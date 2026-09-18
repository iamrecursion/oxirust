//! Chunked-write abstraction for I/O pipelines.
//!
//! Splits a byte stream into fixed-size chunks and optionally applies a
//! per-chunk transform (e.g. checksumming, padding, alignment). Useful for
//! writing media data to disk or network in controlled units.

#![allow(dead_code)]

use std::io::{self, Write};

// ---------------------------------------------------------------------------
// Chunk metadata
// ---------------------------------------------------------------------------

/// Metadata emitted for each completed chunk.
#[derive(Debug, Clone)]
pub struct ChunkInfo {
    /// Zero-based index of this chunk.
    pub index: u64,
    /// Number of payload bytes in this chunk.
    pub payload_len: usize,
    /// Running total of payload bytes written (including this chunk).
    pub cumulative_bytes: u64,
    /// Whether this is the final (possibly short) chunk.
    pub is_final: bool,
}

// ---------------------------------------------------------------------------
// ChunkedWriter
// ---------------------------------------------------------------------------

/// A writer that buffers output into fixed-size chunks before flushing
/// each chunk to the inner writer.
///
/// When the internal buffer reaches `chunk_size`, the chunk is flushed and
/// the optional callback is invoked with [`ChunkInfo`].
pub struct ChunkedWriter<W, F> {
    /// Inner writer.
    inner: W,
    /// Per-chunk callback.
    callback: F,
    /// Buffer accumulating the current chunk.
    buffer: Vec<u8>,
    /// Target chunk size.
    chunk_size: usize,
    /// Number of chunks written.
    chunk_count: u64,
    /// Cumulative payload bytes written.
    total_bytes: u64,
}

impl<W: Write, F: FnMut(&ChunkInfo)> ChunkedWriter<W, F> {
    /// Create a new `ChunkedWriter` with the given chunk size and callback.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is zero.
    pub fn new(inner: W, chunk_size: usize, callback: F) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        Self {
            inner,
            callback,
            buffer: Vec::with_capacity(chunk_size),
            chunk_size,
            chunk_count: 0,
            total_bytes: 0,
        }
    }

    /// Return the number of complete chunks written so far.
    #[must_use]
    pub fn chunk_count(&self) -> u64 {
        self.chunk_count
    }

    /// Return total payload bytes written so far.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Return the configured chunk size.
    #[must_use]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Return how many bytes are buffered in the current (incomplete) chunk.
    #[must_use]
    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    /// Flush the current buffer as the final (possibly short) chunk.
    ///
    /// This should be called after all data has been written to ensure any
    /// remaining bytes are flushed. Calling [`Write::flush`] also triggers
    /// this behavior.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing or flushing the inner writer fails.
    pub fn finish(&mut self) -> io::Result<()> {
        if !self.buffer.is_empty() {
            self.flush_chunk(true)?;
        }
        self.inner.flush()
    }

    /// Consume this writer and return the inner writer. Remaining buffered
    /// data is flushed first.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if flushing the remaining data fails.
    pub fn into_inner(mut self) -> io::Result<W> {
        self.finish()?;
        Ok(self.inner)
    }

    /// Write the buffered chunk out.
    fn flush_chunk(&mut self, is_final: bool) -> io::Result<()> {
        let payload_len = self.buffer.len();
        self.inner.write_all(&self.buffer)?;
        self.total_bytes += payload_len as u64;

        let info = ChunkInfo {
            index: self.chunk_count,
            payload_len,
            cumulative_bytes: self.total_bytes,
            is_final,
        };
        (self.callback)(&info);

        self.chunk_count += 1;
        self.buffer.clear();
        Ok(())
    }
}

impl<W: Write, F: FnMut(&ChunkInfo)> Write for ChunkedWriter<W, F> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut offset = 0;
        while offset < buf.len() {
            let remaining_cap = self.chunk_size - self.buffer.len();
            let to_copy = remaining_cap.min(buf.len() - offset);
            self.buffer
                .extend_from_slice(&buf[offset..offset + to_copy]);
            offset += to_copy;

            if self.buffer.len() >= self.chunk_size {
                self.flush_chunk(false)?;
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.finish()
    }
}

// ---------------------------------------------------------------------------
// AlignedChunkWriter
// ---------------------------------------------------------------------------

/// A writer that pads each chunk to a fixed alignment before writing.
///
/// Useful for writing to block devices or formats that require aligned
/// offsets (e.g. 512-byte or 4096-byte alignment).
pub struct AlignedChunkWriter<W> {
    /// Inner writer.
    inner: W,
    /// Alignment in bytes.
    alignment: usize,
    /// Padding byte value.
    pad_byte: u8,
    /// Buffer for current chunk.
    buffer: Vec<u8>,
    /// Chunks written.
    chunks_written: u64,
}

impl<W: Write> AlignedChunkWriter<W> {
    /// Create a new aligned writer.
    ///
    /// # Panics
    ///
    /// Panics if `alignment` is zero or not a power of two.
    pub fn new(inner: W, alignment: usize) -> Self {
        assert!(
            alignment > 0 && alignment.is_power_of_two(),
            "alignment must be a power of two"
        );
        Self {
            inner,
            alignment,
            pad_byte: 0,
            buffer: Vec::with_capacity(alignment),
            chunks_written: 0,
        }
    }

    /// Set the padding byte (default 0).
    #[must_use]
    pub fn with_pad_byte(mut self, byte: u8) -> Self {
        self.pad_byte = byte;
        self
    }

    /// Return the number of aligned chunks written.
    #[must_use]
    pub fn chunks_written(&self) -> u64 {
        self.chunks_written
    }

    /// Flush the current buffer, padding to alignment.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing or flushing the inner writer fails.
    pub fn finish(&mut self) -> io::Result<()> {
        if !self.buffer.is_empty() {
            let pad_len = self.alignment - (self.buffer.len() % self.alignment);
            if pad_len < self.alignment {
                self.buffer
                    .resize(self.buffer.len() + pad_len, self.pad_byte);
            }
            self.inner.write_all(&self.buffer)?;
            self.chunks_written += 1;
            self.buffer.clear();
        }
        self.inner.flush()
    }
}

impl<W: Write> Write for AlignedChunkWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut offset = 0;
        while offset < buf.len() {
            let remaining = self.alignment - self.buffer.len();
            let to_copy = remaining.min(buf.len() - offset);
            self.buffer
                .extend_from_slice(&buf[offset..offset + to_copy]);
            offset += to_copy;

            if self.buffer.len() >= self.alignment {
                self.inner.write_all(&self.buffer)?;
                self.chunks_written += 1;
                self.buffer.clear();
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.finish()
    }
}

// ---------------------------------------------------------------------------
// CoalescingWriter — batches small writes into larger I/O operations
// ---------------------------------------------------------------------------

/// Statistics tracked by the `CoalescingWriter`.
#[derive(Debug, Clone, Default)]
pub struct CoalesceStats {
    /// Number of `write()` calls received from the caller.
    pub writes_received: u64,
    /// Number of actual write operations issued to the inner writer.
    pub writes_issued: u64,
    /// Total bytes written.
    pub total_bytes: u64,
}

impl CoalesceStats {
    /// Coalescing ratio: how many caller writes were batched per issued write.
    ///
    /// Returns `0.0` if no writes have been issued.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn coalesce_ratio(&self) -> f64 {
        if self.writes_issued == 0 {
            0.0
        } else {
            self.writes_received as f64 / self.writes_issued as f64
        }
    }
}

/// Trigger condition for flushing the coalescing buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoalesceTrigger {
    /// Flush when the buffer reaches at least this many bytes.
    Size(usize),
    /// Flush when this many individual writes have accumulated.
    Count(u64),
    /// Flush when either size or count threshold is reached.
    Either { size: usize, count: u64 },
}

/// A writer that coalesces multiple small writes into larger batches
/// before flushing to the underlying writer.
///
/// This is particularly useful when many small writes (e.g. metadata fields,
/// small NAL units) would otherwise cause excessive I/O syscalls.
pub struct CoalescingWriter<W> {
    inner: W,
    buffer: Vec<u8>,
    trigger: CoalesceTrigger,
    writes_in_buffer: u64,
    stats: CoalesceStats,
}

impl<W: Write> CoalescingWriter<W> {
    /// Create a new coalescing writer with the given trigger.
    pub fn new(inner: W, trigger: CoalesceTrigger) -> Self {
        let initial_cap = match trigger {
            CoalesceTrigger::Size(s) | CoalesceTrigger::Either { size: s, .. } => s,
            CoalesceTrigger::Count(_) => 4096,
        };
        Self {
            inner,
            buffer: Vec::with_capacity(initial_cap),
            trigger,
            writes_in_buffer: 0,
            stats: CoalesceStats::default(),
        }
    }

    /// Return a snapshot of accumulated statistics.
    #[must_use]
    pub fn stats(&self) -> CoalesceStats {
        self.stats.clone()
    }

    /// Return the number of bytes currently buffered.
    #[must_use]
    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    /// Flush the coalescing buffer and the inner writer, then return the
    /// inner writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if flushing fails.
    pub fn into_inner(mut self) -> io::Result<W> {
        self.flush_buffer()?;
        self.inner.flush()?;
        Ok(self.inner)
    }

    /// Flush any buffered data and the inner writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing or flushing fails.
    pub fn finish(&mut self) -> io::Result<()> {
        self.flush_buffer()?;
        self.inner.flush()
    }

    /// Check if the trigger condition is met and flush if so.
    fn maybe_flush(&mut self) -> io::Result<()> {
        let should_flush = match self.trigger {
            CoalesceTrigger::Size(s) => self.buffer.len() >= s,
            CoalesceTrigger::Count(c) => self.writes_in_buffer >= c,
            CoalesceTrigger::Either { size, count } => {
                self.buffer.len() >= size || self.writes_in_buffer >= count
            }
        };
        if should_flush {
            self.flush_buffer()?;
        }
        Ok(())
    }

    /// Flush the internal buffer to the inner writer.
    fn flush_buffer(&mut self) -> io::Result<()> {
        if !self.buffer.is_empty() {
            self.inner.write_all(&self.buffer)?;
            self.stats.writes_issued += 1;
            self.stats.total_bytes += self.buffer.len() as u64;
            self.buffer.clear();
            self.writes_in_buffer = 0;
        }
        Ok(())
    }
}

impl<W: Write> Write for CoalescingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        self.writes_in_buffer += 1;
        self.stats.writes_received += 1;
        self.maybe_flush()?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunked_basic() {
        let mut out = Vec::new();
        let mut infos = Vec::new();
        {
            let mut w = ChunkedWriter::new(&mut out, 4, |info| {
                infos.push(info.clone());
            });
            w.write_all(b"abcdefghij").expect("failed to write");
            w.finish().expect("finish should succeed");
        }
        assert_eq!(out, b"abcdefghij");
        // 4 + 4 = 2 full chunks, plus 2 remaining = 3 chunks
        assert_eq!(infos.len(), 3);
        assert_eq!(infos[0].payload_len, 4);
        assert_eq!(infos[1].payload_len, 4);
        assert_eq!(infos[2].payload_len, 2);
        assert!(infos[2].is_final);
    }

    #[test]
    fn test_chunked_exact_multiple() {
        let mut out = Vec::new();
        let mut count = 0u64;
        {
            let mut w = ChunkedWriter::new(&mut out, 5, |_| count += 1);
            w.write_all(b"12345").expect("failed to write");
            w.finish().expect("finish should succeed");
        }
        assert_eq!(out, b"12345");
        assert_eq!(count, 1); // exactly one full chunk, finish is no-op
    }

    #[test]
    fn test_chunked_empty() {
        let mut out = Vec::new();
        let mut count = 0u64;
        {
            let mut w = ChunkedWriter::new(&mut out, 10, |_| count += 1);
            w.finish().expect("finish should succeed");
        }
        assert!(out.is_empty());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_chunk_info_fields() {
        let mut out = Vec::new();
        let mut infos = Vec::new();
        {
            let mut w = ChunkedWriter::new(&mut out, 3, |info| infos.push(info.clone()));
            w.write_all(b"abcdef").expect("failed to write");
            w.finish().expect("finish should succeed");
        }
        assert_eq!(infos[0].index, 0);
        assert_eq!(infos[0].cumulative_bytes, 3);
        assert_eq!(infos[1].index, 1);
        assert_eq!(infos[1].cumulative_bytes, 6);
    }

    #[test]
    fn test_chunked_total_bytes() {
        let mut out = Vec::new();
        let mut w = ChunkedWriter::new(&mut out, 8, |_| {});
        w.write_all(b"hello world").expect("failed to write");
        w.finish().expect("finish should succeed");
        assert_eq!(w.total_bytes(), 11);
    }

    #[test]
    fn test_chunked_chunk_count() {
        let mut out = Vec::new();
        let mut w = ChunkedWriter::new(&mut out, 4, |_| {});
        w.write_all(b"0123456789ab").expect("failed to write");
        w.finish().expect("finish should succeed");
        assert_eq!(w.chunk_count(), 3);
    }

    #[test]
    fn test_chunked_buffered_len() {
        let mut out = Vec::new();
        let mut w = ChunkedWriter::new(&mut out, 10, |_| {});
        w.write_all(b"abc").expect("failed to write");
        assert_eq!(w.buffered_len(), 3);
    }

    #[test]
    #[should_panic(expected = "chunk_size must be > 0")]
    fn test_chunked_zero_size_panics() {
        let mut out = Vec::new();
        let _w = ChunkedWriter::new(&mut out, 0, |_| {});
    }

    #[test]
    fn test_aligned_basic() {
        let mut out = Vec::new();
        {
            let mut w = AlignedChunkWriter::new(&mut out, 8);
            w.write_all(b"hello").expect("failed to write");
            w.finish().expect("finish should succeed");
        }
        assert_eq!(out.len(), 8); // padded to 8
        assert_eq!(&out[..5], b"hello");
        assert!(out[5..].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_aligned_exact() {
        let mut out = Vec::new();
        {
            let mut w = AlignedChunkWriter::new(&mut out, 4);
            w.write_all(b"abcd").expect("failed to write");
            w.finish().expect("finish should succeed");
        }
        assert_eq!(out, b"abcd");
    }

    #[test]
    fn test_aligned_pad_byte() {
        let mut out = Vec::new();
        {
            let mut w = AlignedChunkWriter::new(&mut out, 8).with_pad_byte(0xFF);
            w.write_all(b"hi").expect("failed to write");
            w.finish().expect("finish should succeed");
        }
        assert_eq!(out.len(), 8);
        assert_eq!(&out[..2], b"hi");
        assert!(out[2..].iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_aligned_chunks_written() {
        let mut out = Vec::new();
        let mut w = AlignedChunkWriter::new(&mut out, 4);
        w.write_all(b"12345678").expect("failed to write"); // two full chunks
        assert_eq!(w.chunks_written(), 2);
    }

    #[test]
    #[should_panic(expected = "alignment must be a power of two")]
    fn test_aligned_non_power_of_two_panics() {
        let mut out = Vec::new();
        let _w = AlignedChunkWriter::new(&mut out, 3);
    }

    // ── CoalescingWriter ─────────────────────────────────────────────────────

    #[test]
    fn test_coalescing_size_trigger() {
        let mut out = Vec::new();
        {
            let mut w = CoalescingWriter::new(&mut out, CoalesceTrigger::Size(10));
            // Write several small chunks
            w.write_all(b"aaa").expect("write");
            w.write_all(b"bbb").expect("write");
            w.write_all(b"cccc").expect("write"); // total 10 => flush
            w.finish().expect("finish");
        }
        assert_eq!(out, b"aaabbbcccc");
    }

    #[test]
    fn test_coalescing_count_trigger() {
        let mut out = Vec::new();
        {
            let mut w = CoalescingWriter::new(&mut out, CoalesceTrigger::Count(3));
            w.write_all(b"a").expect("write");
            w.write_all(b"b").expect("write");
            w.write_all(b"c").expect("write"); // 3 writes => flush
            w.finish().expect("finish");
        }
        assert_eq!(out, b"abc");
    }

    #[test]
    fn test_coalescing_either_trigger() {
        let mut out = Vec::new();
        {
            let mut w = CoalescingWriter::new(
                &mut out,
                CoalesceTrigger::Either {
                    size: 100,
                    count: 2,
                },
            );
            w.write_all(b"x").expect("write");
            w.write_all(b"y").expect("write"); // count=2 triggers flush
            w.finish().expect("finish");
        }
        assert_eq!(out, b"xy");
    }

    #[test]
    fn test_coalescing_stats() {
        let mut out = Vec::new();
        let mut w = CoalescingWriter::new(&mut out, CoalesceTrigger::Size(100));
        // Write 5 small chunks that don't trigger flush
        for _ in 0..5 {
            w.write_all(b"hi").expect("write");
        }
        w.finish().expect("finish");

        let stats = w.stats();
        assert_eq!(stats.writes_received, 5);
        assert_eq!(stats.writes_issued, 1); // one coalesced write
        assert_eq!(stats.total_bytes, 10);
        assert!((stats.coalesce_ratio() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_coalescing_empty_finish() {
        let mut out = Vec::new();
        {
            let mut w = CoalescingWriter::new(&mut out, CoalesceTrigger::Size(10));
            w.finish().expect("finish");
            assert_eq!(w.stats().writes_issued, 0);
        }
        assert!(out.is_empty());
    }

    #[test]
    fn test_coalescing_large_write_triggers_immediately() {
        let mut out = Vec::new();
        {
            let mut w = CoalescingWriter::new(&mut out, CoalesceTrigger::Size(5));
            w.write_all(b"abcdefgh").expect("write"); // 8 > 5 => flush
            let stats = w.stats();
            assert_eq!(stats.writes_issued, 1);
            w.finish().expect("finish");
        }
        assert_eq!(out, b"abcdefgh");
    }

    #[test]
    fn test_coalescing_buffered_len() {
        let mut out = Vec::new();
        let mut w = CoalescingWriter::new(&mut out, CoalesceTrigger::Size(100));
        w.write_all(b"hello").expect("write");
        assert_eq!(w.buffered_len(), 5);
        w.finish().expect("finish");
        assert_eq!(w.buffered_len(), 0);
    }

    #[test]
    fn test_coalescing_into_inner() {
        let out = Vec::new();
        let mut w = CoalescingWriter::new(out, CoalesceTrigger::Size(100));
        w.write_all(b"data").expect("write");
        let result = w.into_inner().expect("into_inner");
        assert_eq!(result, b"data");
    }

    #[test]
    fn test_coalescing_ratio_zero_when_empty() {
        let stats = CoalesceStats::default();
        assert_eq!(stats.coalesce_ratio(), 0.0);
    }

    #[test]
    fn test_coalescing_multiple_flushes() {
        let mut out = Vec::new();
        {
            let mut w = CoalescingWriter::new(&mut out, CoalesceTrigger::Size(4));
            w.write_all(b"aaaa").expect("write"); // triggers flush (4 >= 4)
            w.write_all(b"bbbb").expect("write"); // triggers flush
            w.write_all(b"cc").expect("write"); // below threshold
            w.finish().expect("finish");
            assert_eq!(w.stats().writes_issued, 3); // 2 triggered + 1 finish
        }
        assert_eq!(out, b"aaaabbbbcc");
    }
}
