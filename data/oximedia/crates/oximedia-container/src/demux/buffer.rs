//! Packet buffering and read-ahead I/O for demuxers.
//!
//! Provides:
//! - [`PacketBuffer`] — per-stream packet queues with seek support and
//!   configurable maximum buffer depths.
//! - [`ReadAheadBuffer`] — a sequential read-ahead I/O buffer that fills a
//!   large internal ring buffer from any [`std::io::Read`] source, amortising
//!   small read calls and tracking hit/miss statistics.

use std::collections::{HashMap, VecDeque};

/// A buffered packet for a single stream.
#[derive(Debug, Clone)]
pub struct BufferedPacket {
    /// Stream index this packet belongs to.
    pub stream_index: usize,
    /// Presentation timestamp.
    pub pts: i64,
    /// Optional decode timestamp.
    pub dts: Option<i64>,
    /// Raw compressed data.
    pub data: Vec<u8>,
    /// Whether this packet contains a keyframe.
    pub is_keyframe: bool,
    /// Duration in stream timebase units.
    pub duration: Option<i64>,
}

impl BufferedPacket {
    /// Create a new buffered packet.
    #[must_use]
    pub fn new(stream_index: usize, pts: i64, data: Vec<u8>, is_keyframe: bool) -> Self {
        Self {
            stream_index,
            pts,
            dts: None,
            data,
            is_keyframe,
            duration: None,
        }
    }
}

/// Per-stream packet queue.
struct StreamQueue {
    queue: VecDeque<BufferedPacket>,
    max_depth: usize,
}

impl StreamQueue {
    fn new(max_depth: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            max_depth: max_depth.max(1),
        }
    }

    /// Push a packet; returns evicted packet if the queue was full.
    fn push(&mut self, pkt: BufferedPacket) -> Option<BufferedPacket> {
        let evicted = if self.queue.len() >= self.max_depth {
            self.queue.pop_front()
        } else {
            None
        };
        self.queue.push_back(pkt);
        evicted
    }

    fn pop(&mut self) -> Option<BufferedPacket> {
        self.queue.pop_front()
    }

    fn peek(&self) -> Option<&BufferedPacket> {
        self.queue.front()
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn clear(&mut self) {
        self.queue.clear();
    }

    /// Discard all packets whose PTS < `target_pts`.
    fn discard_before(&mut self, target_pts: i64) {
        while let Some(front) = self.queue.front() {
            if front.pts < target_pts {
                self.queue.pop_front();
            } else {
                break;
            }
        }
    }

    /// Find the position of the first keyframe at or after `target_pts`.
    fn first_keyframe_pos_from(&self, target_pts: i64) -> Option<usize> {
        self.queue
            .iter()
            .enumerate()
            .find(|(_, p)| p.pts >= target_pts && p.is_keyframe)
            .map(|(i, _)| i)
    }
}

/// Multi-stream packet buffer supporting seek operations.
pub struct PacketBuffer {
    streams: HashMap<usize, StreamQueue>,
    default_depth: usize,
}

impl PacketBuffer {
    /// Create a new buffer with the given per-stream maximum depth.
    #[must_use]
    pub fn new(default_depth: usize) -> Self {
        Self {
            streams: HashMap::new(),
            default_depth,
        }
    }

    /// Push a packet into the appropriate stream queue.
    ///
    /// Returns any evicted packet (when the queue was at capacity).
    pub fn push(&mut self, pkt: BufferedPacket) -> Option<BufferedPacket> {
        let depth = self.default_depth;
        self.streams
            .entry(pkt.stream_index)
            .or_insert_with(|| StreamQueue::new(depth))
            .push(pkt)
    }

    /// Pop the next packet from a specific stream.
    pub fn pop_stream(&mut self, stream_index: usize) -> Option<BufferedPacket> {
        self.streams.get_mut(&stream_index)?.pop()
    }

    /// Pop the packet with the lowest PTS across all streams.
    pub fn pop_lowest_pts(&mut self) -> Option<BufferedPacket> {
        let stream_idx = self
            .streams
            .iter()
            .filter_map(|(idx, q)| q.peek().map(|p| (*idx, p.pts)))
            .min_by_key(|&(_, pts)| pts)
            .map(|(idx, _)| idx)?;
        self.streams.get_mut(&stream_idx)?.pop()
    }

    /// Flush all buffers (e.g., after a seek).
    pub fn flush(&mut self) {
        for q in self.streams.values_mut() {
            q.clear();
        }
    }

    /// Perform a seek: discard packets before `target_pts` on all streams.
    pub fn seek_to_pts(&mut self, target_pts: i64) {
        for q in self.streams.values_mut() {
            q.discard_before(target_pts);
        }
    }

    /// Total number of buffered packets across all streams.
    #[must_use]
    pub fn total_buffered(&self) -> usize {
        self.streams.values().map(StreamQueue::len).sum()
    }

    /// Number of packets buffered for a specific stream.
    #[must_use]
    pub fn stream_depth(&self, stream_index: usize) -> usize {
        self.streams.get(&stream_index).map_or(0, StreamQueue::len)
    }

    /// Returns `true` if all stream queues are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.streams.values().all(StreamQueue::is_empty)
    }

    /// Find the PTS of the first keyframe at or after `target_pts` on the
    /// given stream, without consuming any packets.
    #[must_use]
    pub fn first_keyframe_pts(&self, stream_index: usize, target_pts: i64) -> Option<i64> {
        let q = self.streams.get(&stream_index)?;
        let pos = q.first_keyframe_pos_from(target_pts)?;
        q.queue.get(pos).map(|p| p.pts)
    }
}

impl Default for PacketBuffer {
    fn default() -> Self {
        Self::new(512)
    }
}

// ─── ReadAheadBuffer ─────────────────────────────────────────────────────────

/// Default read-ahead buffer size: 256 KiB.
pub const DEFAULT_READ_AHEAD_SIZE: usize = 256 * 1024;

/// Accumulated I/O statistics for a [`ReadAheadBuffer`].
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    /// Number of read requests satisfied entirely from the internal buffer
    /// (no underlying `Read` call needed).
    pub cache_hits: u64,
    /// Number of read requests that required filling (or partially filling)
    /// the internal buffer from the underlying source.
    pub cache_misses: u64,
    /// Total bytes fetched from the underlying source during fill operations.
    pub bytes_read_ahead: u64,
}

/// A sequential read-ahead I/O buffer.
///
/// `ReadAheadBuffer` wraps any [`std::io::Read`] source.  On each cache miss
/// it fills a large internal buffer (`read_ahead_size` bytes) from the source
/// so that subsequent reads are served from memory.  This amortises the
/// per-call overhead of small reads (e.g. reading fixed-size headers in a
/// tight loop) and gives the kernel a chance to issue efficient large reads.
///
/// # Example
///
/// ```no_run
/// use oximedia_container::demux::buffer::{ReadAheadBuffer, DEFAULT_READ_AHEAD_SIZE};
/// use std::io::Read;
///
/// let file = std::fs::File::open("video.mkv").expect("open");
/// let mut rab = ReadAheadBuffer::new(DEFAULT_READ_AHEAD_SIZE);
/// let mut header = [0u8; 16];
/// let mut src = Box::new(file) as Box<dyn Read>;
/// rab.read(&mut *src, &mut header).expect("read header");
/// println!("hit_rate: {:.1}%", rab.hit_rate() * 100.0);
/// ```
pub struct ReadAheadBuffer {
    /// Internal ring buffer: bytes at `[read_pos, fill_pos)` are valid data.
    buf: Vec<u8>,
    /// Next byte to hand to the caller.
    read_pos: usize,
    /// One past the last valid byte in `buf`.
    fill_pos: usize,
    /// Number of bytes to fetch from the source per fill operation.
    read_ahead_size: usize,
    /// Accumulated statistics.
    stats: BufferStats,
}

impl ReadAheadBuffer {
    /// Creates a new `ReadAheadBuffer` with the given fill size.
    ///
    /// `read_ahead_size` controls how many bytes are fetched from the
    /// underlying source on each cache miss.  Use
    /// [`DEFAULT_READ_AHEAD_SIZE`] (256 KiB) for most workloads.
    #[must_use]
    pub fn new(read_ahead_size: usize) -> Self {
        let size = read_ahead_size.max(1);
        Self {
            buf: vec![0u8; size],
            read_pos: 0,
            fill_pos: 0,
            read_ahead_size: size,
            stats: BufferStats::default(),
        }
    }

    /// Returns the number of bytes currently available in the buffer without
    /// reading from the source.
    #[inline]
    fn available(&self) -> usize {
        self.fill_pos.saturating_sub(self.read_pos)
    }

    /// Refill the internal buffer from `src`.
    ///
    /// Copies any unread tail bytes to the front of the buffer, then reads
    /// up to `read_ahead_size` bytes from `src` into the remaining space.
    fn refill(&mut self, src: &mut dyn std::io::Read) -> std::io::Result<()> {
        // Shift leftover bytes to the front.
        let leftover = self.fill_pos - self.read_pos;
        if leftover > 0 {
            self.buf.copy_within(self.read_pos..self.fill_pos, 0);
        }
        self.read_pos = 0;
        self.fill_pos = leftover;

        // Fill the rest of the buffer.
        let space = self.buf.len() - self.fill_pos;
        if space == 0 {
            return Ok(());
        }
        let n = src.read(&mut self.buf[self.fill_pos..self.fill_pos + space])?;
        self.fill_pos += n;
        self.stats.bytes_read_ahead = self.stats.bytes_read_ahead.saturating_add(n as u64);
        Ok(())
    }

    /// Reads up to `buf.len()` bytes into `buf`, refilling from `src` if
    /// necessary.
    ///
    /// Returns the number of bytes actually written into `buf` (which may be
    /// less than `buf.len()` at end-of-source or if the source returns fewer
    /// bytes than requested).
    ///
    /// # Errors
    ///
    /// Propagates any I/O error returned by the underlying source during a
    /// refill.
    pub fn read(&mut self, src: &mut dyn std::io::Read, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // Fast path: buffer already has enough data.
        if self.available() >= buf.len() {
            let n = buf.len();
            buf.copy_from_slice(&self.buf[self.read_pos..self.read_pos + n]);
            self.read_pos += n;
            self.stats.cache_hits = self.stats.cache_hits.saturating_add(1);
            return Ok(n);
        }

        // Slow path: need a refill.
        self.stats.cache_misses = self.stats.cache_misses.saturating_add(1);
        self.refill(src)?;

        let n = buf.len().min(self.available());
        buf[..n].copy_from_slice(&self.buf[self.read_pos..self.read_pos + n]);
        self.read_pos += n;
        Ok(n)
    }

    /// Returns the configured read-ahead fill size in bytes.
    #[must_use]
    #[inline]
    pub fn read_ahead_size(&self) -> usize {
        self.read_ahead_size
    }

    /// Returns a reference to the accumulated I/O statistics.
    #[must_use]
    pub fn stats(&self) -> &BufferStats {
        &self.stats
    }

    /// Returns the fraction of reads satisfied from the internal buffer.
    ///
    /// The value is in `[0.0, 1.0]`.  Returns `0.0` before any reads have
    /// been performed (to avoid division by zero).
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.cache_hits + self.stats.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.stats.cache_hits as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pkt(stream: usize, pts: i64, keyframe: bool) -> BufferedPacket {
        BufferedPacket::new(stream, pts, vec![0u8; 8], keyframe)
    }

    #[test]
    fn test_buffered_packet_new() {
        let p = make_pkt(0, 1000, true);
        assert_eq!(p.stream_index, 0);
        assert_eq!(p.pts, 1000);
        assert!(p.is_keyframe);
        assert_eq!(p.data.len(), 8);
    }

    #[test]
    fn test_push_and_pop_stream() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 100, true));
        buf.push(make_pkt(0, 200, false));
        let p = buf.pop_stream(0).expect("operation should succeed");
        assert_eq!(p.pts, 100);
        assert_eq!(buf.stream_depth(0), 1);
    }

    #[test]
    fn test_pop_stream_empty() {
        let mut buf = PacketBuffer::new(16);
        assert!(buf.pop_stream(0).is_none());
    }

    #[test]
    fn test_pop_lowest_pts() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 300, false));
        buf.push(make_pkt(1, 100, true));
        buf.push(make_pkt(1, 200, false));

        let p = buf.pop_lowest_pts().expect("operation should succeed");
        assert_eq!(p.pts, 100);
        assert_eq!(p.stream_index, 1);
    }

    #[test]
    fn test_flush() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 100, true));
        buf.push(make_pkt(1, 200, false));
        assert_eq!(buf.total_buffered(), 2);
        buf.flush();
        assert_eq!(buf.total_buffered(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_seek_to_pts() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 50, true));
        buf.push(make_pkt(0, 150, false));
        buf.push(make_pkt(0, 250, false));
        buf.seek_to_pts(100);
        // Packet at 50 should be discarded
        let p = buf.pop_stream(0).expect("operation should succeed");
        assert_eq!(p.pts, 150);
    }

    #[test]
    fn test_eviction_on_overflow() {
        let mut buf = PacketBuffer::new(3);
        buf.push(make_pkt(0, 10, true));
        buf.push(make_pkt(0, 20, false));
        buf.push(make_pkt(0, 30, false));
        // This push should evict pts=10
        let evicted = buf.push(make_pkt(0, 40, false));
        assert!(evicted.is_some());
        assert_eq!(evicted.expect("operation should succeed").pts, 10);
        assert_eq!(buf.stream_depth(0), 3);
    }

    #[test]
    fn test_total_buffered() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 100, false));
        buf.push(make_pkt(1, 200, false));
        buf.push(make_pkt(2, 300, false));
        assert_eq!(buf.total_buffered(), 3);
    }

    #[test]
    fn test_is_empty_initially() {
        let buf = PacketBuffer::new(16);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_first_keyframe_pts() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 0, true));
        buf.push(make_pkt(0, 100, false));
        buf.push(make_pkt(0, 200, true));
        buf.push(make_pkt(0, 300, false));

        // Looking for keyframe >= 150: should find 200
        let kf = buf.first_keyframe_pts(0, 150);
        assert_eq!(kf, Some(200));
    }

    #[test]
    fn test_first_keyframe_pts_none() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 0, false));
        buf.push(make_pkt(0, 100, false));
        let kf = buf.first_keyframe_pts(0, 0);
        assert!(kf.is_none()); // No keyframe in queue
    }

    #[test]
    fn test_default_buffer() {
        let buf: PacketBuffer = Default::default();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_multi_stream_interleave() {
        let mut buf = PacketBuffer::new(64);
        // Video stream
        for i in [0i64, 40, 80, 120] {
            buf.push(make_pkt(0, i, i == 0));
        }
        // Audio stream
        for i in [0i64, 20, 40, 60, 80] {
            buf.push(make_pkt(1, i, true));
        }
        let mut last_pts = -1i64;
        while let Some(p) = buf.pop_lowest_pts() {
            assert!(p.pts >= last_pts, "out of order");
            last_pts = p.pts;
        }
    }

    // ── ReadAheadBuffer tests ─────────────────────────────────────────────

    use std::io::{Cursor, Write};

    /// Write `n` bytes (0x00..0xFF cycling) to a temp file and return the path.
    fn make_sequential_temp_file(n: usize) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        path.push(format!(
            "oximedia_rab_test_{}_{}.bin",
            std::process::id(),
            nanos,
        ));
        let data: Vec<u8> = (0..n).map(|i| (i & 0xFF) as u8).collect();
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(&data).expect("write temp file");
        f.flush().expect("flush");
        path
    }

    #[test]
    fn test_read_ahead_sequential() {
        // Write 64 KiB of data. Use a 4 KiB read-ahead buffer.
        // After the first miss (fills 4 KiB), subsequent small reads (16 bytes
        // each) should be hits → hit_rate > 0.5.
        let total = 64 * 1024usize;
        let path = make_sequential_temp_file(total);
        let mut file = std::fs::File::open(&path).expect("open");
        let mut rab = ReadAheadBuffer::new(4 * 1024);

        let mut small_buf = [0u8; 16];
        let mut bytes_read = 0usize;
        while bytes_read < total {
            let n = rab
                .read(&mut file, &mut small_buf)
                .expect("read should not fail");
            if n == 0 {
                break;
            }
            bytes_read += n;
        }

        let rate = rab.hit_rate();
        assert!(
            rate > 0.5,
            "hit_rate {rate} should be > 0.5 after warmup (reads: {bytes_read})",
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_ahead_stats() {
        // 100 reads of 1 byte each from an in-memory cursor; after warmup the
        // stats.bytes_read_ahead must be > 0 (the buffer fetched ahead).
        let data: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let mut src = Cursor::new(data.clone());
        let mut rab = ReadAheadBuffer::new(512);
        let mut one = [0u8; 1];
        for _ in 0..100 {
            let n = rab.read(&mut src, &mut one).expect("read ok");
            if n == 0 {
                break;
            }
        }
        assert!(
            rab.stats().bytes_read_ahead > 0,
            "bytes_read_ahead should be > 0 after reads",
        );
        assert!(
            rab.stats().cache_hits + rab.stats().cache_misses >= 100,
            "total reads should be >= 100",
        );
    }

    #[test]
    fn test_read_ahead_correctness() {
        // Verify data integrity: all bytes read must match the source pattern.
        let data: Vec<u8> = (0u8..=255).cycle().take(2048).collect();
        let mut src = Cursor::new(data.clone());
        let mut rab = ReadAheadBuffer::new(256);
        let mut result = Vec::new();
        let mut chunk = [0u8; 37]; // prime-sized reads to stress alignment
        loop {
            let n = rab.read(&mut src, &mut chunk).expect("read ok");
            if n == 0 {
                break;
            }
            result.extend_from_slice(&chunk[..n]);
        }
        assert_eq!(result, data, "read bytes must exactly match source");
    }

    #[test]
    fn test_hit_rate_zero_before_reads() {
        let rab = ReadAheadBuffer::new(1024);
        assert_eq!(rab.hit_rate(), 0.0, "hit_rate must be 0.0 before any reads");
    }
}
