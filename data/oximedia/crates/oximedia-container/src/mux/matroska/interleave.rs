//! Parallel multi-stream interleaving for Matroska muxing.
//!
//! This module provides [`ParallelInterleaver`], which accepts batches of
//! packets from multiple tracks and — when the batch is large enough — uses
//! `std::thread::scope` to serialise the per-packet EBML byte encoding in
//! parallel before writing the results sequentially to the underlying sink.
//!
//! # Why Parallel?
//!
//! Serialising a `SimpleBlock` is CPU-bound work: it involves encoding the
//! track VINT, copying the sample payload into an `Vec<u8>`, and writing EBML
//! size headers.  For HD/UHD workflows with many simultaneous tracks (video +
//! audio + subtitle) the encoding of independent blocks is embarrassingly
//! parallel.  The actual **write** to the sink must remain sequential because
//! most writers (`FileSource`, `MemorySource`) are not thread-safe.
//!
//! # Behaviour
//!
//! The interleaver buffers up to `max_pending` packets.  Once the buffer is
//! full (or [`flush`](ParallelInterleaver::flush) is called explicitly) it:
//!
//! 1. Sorts the batch by PTS so the output stream is correctly interleaved.
//! 2. Encodes each packet's `SimpleBlock` bytes in parallel using
//!    [`std::thread::scope`].
//! 3. Returns the encoded blobs in presentation order so the caller can write
//!    them to the sink one by one.
//!
//! # Integration
//!
//! Call [`push`](ParallelInterleaver::push) for each incoming packet.  The
//! return value contains encoded blobs that are ready to write; an empty
//! `Vec` means the packet was buffered and no output is ready yet.
//!
//! Call [`flush`](ParallelInterleaver::flush) at stream end to drain any
//! remaining buffered packets.

#![forbid(unsafe_code)]

use oximedia_core::Rational;

use crate::{Packet, StreamInfo};

// ─── Types ──────────────────────────────────────────────────────────────────

/// An encoded, write-ready Matroska `SimpleBlock` blob together with the
/// track number and timecode metadata needed by the cluster writer.
#[derive(Debug, Clone)]
pub struct EncodedBlock {
    /// 1-based track number.
    pub track_num: u64,
    /// Absolute timecode in cluster-timescale units.
    pub abs_timecode: i64,
    /// Relative timecode (abs_timecode – cluster_timecode), clamped to i16.
    pub rel_timecode: i16,
    /// `true` if this block contains a keyframe.
    pub is_keyframe: bool,
    /// Complete EBML `SimpleBlock` payload (track VINT + rel-timecode + flags +
    /// sample data), *without* the outer EBML element header.  The caller must
    /// prepend the `SimpleBlock` element ID and size.
    pub payload: Vec<u8>,
    /// Stream index from the original [`Packet`].
    pub stream_index: usize,
    /// Byte size of the raw sample data (before EBML wrapping).
    pub data_len: usize,
}

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for [`ParallelInterleaver`].
#[derive(Debug, Clone)]
pub struct InterleaverConfig {
    /// Maximum number of packets to buffer before flushing.
    ///
    /// A larger value increases parallelism at the cost of memory and latency.
    /// Default: 32.
    pub max_pending: usize,

    /// Minimum number of packets required to trigger parallel encoding.
    ///
    /// Below this threshold the interleaver encodes sequentially to avoid
    /// thread-spawn overhead.  Default: 4.
    pub parallel_threshold: usize,
}

impl Default for InterleaverConfig {
    fn default() -> Self {
        Self {
            max_pending: 32,
            parallel_threshold: 4,
        }
    }
}

impl InterleaverConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_pending: 32,
            parallel_threshold: 4,
        }
    }

    /// Sets the maximum pending packet count.
    #[must_use]
    pub const fn with_max_pending(mut self, n: usize) -> Self {
        self.max_pending = n;
        self
    }

    /// Sets the parallel encoding threshold.
    #[must_use]
    pub const fn with_parallel_threshold(mut self, n: usize) -> Self {
        self.parallel_threshold = n;
        self
    }
}

// ─── Internal pending packet ─────────────────────────────────────────────────

/// A buffered packet waiting for batch encoding.
#[derive(Clone)]
struct Pending {
    packet: Packet,
    track_num: u64,
    abs_timecode: i64,
    cluster_timecode: i64,
}

// ─── ParallelInterleaver ─────────────────────────────────────────────────────

/// Buffers packets from multiple tracks and encodes them in parallel batches.
///
/// See [module-level docs](self) for detailed description.
pub struct ParallelInterleaver {
    config: InterleaverConfig,
    pending: Vec<Pending>,
}

impl ParallelInterleaver {
    /// Creates a new interleaver with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(InterleaverConfig::default())
    }

    /// Creates a new interleaver with the given configuration.
    #[must_use]
    pub fn with_config(config: InterleaverConfig) -> Self {
        Self {
            config,
            pending: Vec::with_capacity(config.max_pending),
        }
    }

    /// Returns the number of packets currently buffered.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Pushes a packet into the buffer.
    ///
    /// If the buffer has reached `max_pending`, the batch is encoded and the
    /// result is returned.  Otherwise an empty `Vec` is returned (the packet
    /// is buffered).
    ///
    /// # Arguments
    ///
    /// * `packet`          – The incoming media packet.
    /// * `stream_info`     – Stream metadata (codec, timebase).
    /// * `track_num`       – 1-based Matroska track number for this stream.
    /// * `cluster_timecode`– Timecode of the current cluster (in mux units).
    /// * `to_timecode`     – Closure converting `(pts, timebase) → i64`.
    pub fn push<F>(
        &mut self,
        packet: Packet,
        stream_info: &StreamInfo,
        track_num: u64,
        cluster_timecode: i64,
        to_timecode: F,
    ) -> Vec<EncodedBlock>
    where
        F: Fn(i64, Rational) -> i64,
    {
        let abs_timecode = to_timecode(packet.pts(), stream_info.timebase);
        self.pending.push(Pending {
            packet,
            track_num,
            abs_timecode,
            cluster_timecode,
        });

        if self.pending.len() >= self.config.max_pending {
            self.flush_inner()
        } else {
            Vec::new()
        }
    }

    /// Forces encoding of all buffered packets and returns the results.
    ///
    /// Always call this when the stream ends (or at cluster boundaries) to
    /// drain the buffer.
    pub fn flush(&mut self) -> Vec<EncodedBlock> {
        self.flush_inner()
    }

    // ── Internal ──────────────────────────────────────────────────────────

    fn flush_inner(&mut self) -> Vec<EncodedBlock> {
        if self.pending.is_empty() {
            return Vec::new();
        }

        // Sort by PTS for correct interleaving order.
        self.pending
            .sort_unstable_by_key(|p| p.abs_timecode);

        let items: Vec<Pending> = std::mem::take(&mut self.pending);
        let n = items.len();

        if n < self.config.parallel_threshold {
            // Below threshold: encode sequentially to avoid thread overhead.
            items.into_iter().map(encode_pending).collect()
        } else {
            // Above threshold: encode in parallel using scoped threads.
            parallel_encode(items)
        }
    }
}

impl Default for ParallelInterleaver {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Encoding helpers ────────────────────────────────────────────────────────

/// Encodes a single pending packet into an [`EncodedBlock`].
fn encode_pending(p: Pending) -> EncodedBlock {
    let is_keyframe = p.packet.is_keyframe();
    let data = &p.packet.data;
    let data_len = data.len();

    // Relative timecode: clamped to i16 range.
    let rel = (p.abs_timecode - p.cluster_timecode).clamp(i16::MIN as i64, i16::MAX as i64) as i16;

    // Build SimpleBlock payload (without outer EBML header):
    //   track VINT | rel-timecode (2 bytes BE) | flags (1 byte) | sample data
    let track_vint = encode_vint(p.track_num);
    let mut payload = Vec::with_capacity(track_vint.len() + 3 + data_len);
    payload.extend_from_slice(&track_vint);
    payload.extend_from_slice(&rel.to_be_bytes());
    payload.push(if is_keyframe { 0x80 } else { 0x00 });
    payload.extend_from_slice(data);

    EncodedBlock {
        track_num: p.track_num,
        abs_timecode: p.abs_timecode,
        rel_timecode: rel,
        is_keyframe,
        payload,
        stream_index: p.packet.stream_index,
        data_len,
    }
}

/// Encodes multiple packets in parallel using scoped threads.
///
/// Each element of `items` is encoded independently on its own thread; the
/// results are collected in order and returned.
fn parallel_encode(items: Vec<Pending>) -> Vec<EncodedBlock> {
    let n = items.len();
    let mut results: Vec<Option<EncodedBlock>> = (0..n).map(|_| None).collect();

    std::thread::scope(|s| {
        // Pair each pending item with a mutable reference to its output slot.
        let pairs: Vec<(&Pending, &mut Option<EncodedBlock>)> =
            items.iter().zip(results.iter_mut()).collect();

        // Spawn one thread per (item, output) pair.
        // `std::thread::scope` guarantees all threads finish before the scope
        // exits, so the `&mut Option<…>` borrows are safe.
        let handles: Vec<_> = pairs
            .into_iter()
            .map(|(item, slot)| {
                s.spawn(move || {
                    *slot = Some(encode_pending(item.clone()));
                })
            })
            .collect();

        // Join all threads (the scope ensures this before it exits).
        for h in handles {
            let _ = h.join();
        }
    });

    results.into_iter().flatten().collect()
}

/// Encodes a VINT (variable-length integer) for a Matroska track number.
///
/// Matroska VINTs set a leading 1-bit marker at the position determined by the
/// number of bytes required:
/// - values 0–126:   1 byte  (`0x80 | value`)
/// - values 127–16382:  2 bytes
/// - values 16383–2097150: 3 bytes
/// - values 2097151–268435454: 4 bytes
fn encode_vint(value: u64) -> Vec<u8> {
    if value < 0x80 {
        vec![0x80 | value as u8]
    } else if value < 0x4000 {
        vec![0x40 | (value >> 8) as u8, value as u8]
    } else if value < 0x20_0000 {
        vec![0x20 | (value >> 16) as u8, (value >> 8) as u8, value as u8]
    } else if value < 0x1000_0000 {
        vec![
            0x10 | (value >> 24) as u8,
            (value >> 16) as u8,
            (value >> 8) as u8,
            value as u8,
        ]
    } else {
        // 8-byte fallback for very large values (should never occur for track IDs)
        vec![
            0x01,
            (value >> 48) as u8,
            (value >> 40) as u8,
            (value >> 32) as u8,
            (value >> 24) as u8,
            (value >> 16) as u8,
            (value >> 8) as u8,
            value as u8,
        ]
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use oximedia_core::{CodecId, Rational, Timestamp};

    use crate::{PacketFlags, StreamInfo};

    fn make_packet(stream_index: usize, pts: i64, keyframe: bool, size: usize) -> Packet {
        let flags = if keyframe {
            PacketFlags::KEYFRAME
        } else {
            PacketFlags::empty()
        };
        let data = Bytes::from(vec![0xABu8; size]);
        let ts = Timestamp::new(pts, Rational::new(1, 1000));
        Packet::new(stream_index, data, ts, flags)
    }

    fn make_stream(index: usize) -> StreamInfo {
        StreamInfo::new(index, CodecId::Vp9, Rational::new(1, 1000))
    }

    fn identity_timecode(pts: i64, _timebase: Rational) -> i64 {
        pts
    }

    // ── encode_vint ──────────────────────────────────────────────────────

    #[test]
    fn test_encode_vint_1byte() {
        assert_eq!(encode_vint(1), vec![0x81]);
        assert_eq!(encode_vint(0), vec![0x80]);
        assert_eq!(encode_vint(127), vec![0xFF]);
    }

    #[test]
    fn test_encode_vint_2byte() {
        assert_eq!(encode_vint(128), vec![0x40, 0x80]);
    }

    // ── encode_pending ───────────────────────────────────────────────────

    #[test]
    fn test_encode_pending_keyframe_flag() {
        let pkt = make_packet(0, 1000, true, 10);
        let p = Pending {
            packet: pkt,
            track_num: 1,
            abs_timecode: 1000,
            cluster_timecode: 0,
        };
        let block = encode_pending(p);
        assert!(block.is_keyframe);
        // flags byte: 0x80 for keyframe
        // layout: vint(1) = [0x81] | rel [0x03 0xE8] | flags [0x80] | data (10 bytes)
        let flags_offset = encode_vint(1).len() + 2;
        assert_eq!(block.payload[flags_offset], 0x80);
    }

    #[test]
    fn test_encode_pending_non_keyframe() {
        let pkt = make_packet(0, 0, false, 5);
        let p = Pending {
            packet: pkt,
            track_num: 1,
            abs_timecode: 0,
            cluster_timecode: 0,
        };
        let block = encode_pending(p);
        assert!(!block.is_keyframe);
        let flags_offset = encode_vint(1).len() + 2;
        assert_eq!(block.payload[flags_offset], 0x00);
    }

    #[test]
    fn test_encode_pending_rel_timecode() {
        let pkt = make_packet(0, 5000, false, 4);
        let p = Pending {
            packet: pkt,
            track_num: 2,
            abs_timecode: 5000,
            cluster_timecode: 4000,
        };
        let block = encode_pending(p);
        // rel = 5000 - 4000 = 1000 = 0x03E8
        assert_eq!(block.rel_timecode, 1000);
        let vint_len = encode_vint(2).len();
        let rel_bytes = &block.payload[vint_len..vint_len + 2];
        assert_eq!(rel_bytes, &[0x03, 0xE8]);
    }

    #[test]
    fn test_encode_pending_data_appended() {
        let pkt = make_packet(0, 0, true, 8);
        let p = Pending {
            packet: pkt,
            track_num: 1,
            abs_timecode: 0,
            cluster_timecode: 0,
        };
        let block = encode_pending(p);
        assert_eq!(block.data_len, 8);
        // payload length = vint(1)=1 + rel(2) + flags(1) + data(8) = 12
        assert_eq!(block.payload.len(), 1 + 2 + 1 + 8);
    }

    // ── ParallelInterleaver ──────────────────────────────────────────────

    #[test]
    fn test_interleaver_buffers_below_threshold() {
        let config = InterleaverConfig::new().with_max_pending(8);
        let mut il = ParallelInterleaver::with_config(config);
        let stream = make_stream(0);

        let pkt = make_packet(0, 0, true, 4);
        let result = il.push(pkt, &stream, 1, 0, identity_timecode);
        assert!(result.is_empty(), "should buffer, not encode yet");
        assert_eq!(il.pending_count(), 1);
    }

    #[test]
    fn test_interleaver_encodes_when_full() {
        let config = InterleaverConfig::new()
            .with_max_pending(4)
            .with_parallel_threshold(2);
        let mut il = ParallelInterleaver::with_config(config);
        let stream = make_stream(0);

        let mut last_result = Vec::new();
        for i in 0..4u64 {
            let pkt = make_packet(0, i as i64 * 100, i == 0, 4);
            last_result = il.push(pkt, &stream, 1, 0, identity_timecode);
        }

        // On 4th push (= max_pending) should return 4 encoded blocks
        assert_eq!(last_result.len(), 4);
        assert_eq!(il.pending_count(), 0);
    }

    #[test]
    fn test_interleaver_flush_drains_buffer() {
        let mut il = ParallelInterleaver::new();
        let stream = make_stream(0);

        for i in 0..3u64 {
            let pkt = make_packet(0, i as i64 * 100, i == 0, 4);
            il.push(pkt, &stream, 1, 0, identity_timecode);
        }

        assert_eq!(il.pending_count(), 3);
        let blocks = il.flush();
        assert_eq!(blocks.len(), 3);
        assert_eq!(il.pending_count(), 0);
    }

    #[test]
    fn test_interleaver_flush_empty_returns_empty() {
        let mut il = ParallelInterleaver::new();
        let blocks = il.flush();
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_interleaver_sorts_by_pts() {
        let config = InterleaverConfig::new()
            .with_max_pending(3)
            .with_parallel_threshold(2);
        let mut il = ParallelInterleaver::with_config(config);
        let stream = make_stream(0);

        // Push out-of-order: 300, 100, 200 → should come back as 100, 200, 300
        let pts_values: &[i64] = &[300, 100, 200];
        let mut result = Vec::new();
        for &pts in pts_values {
            let pkt = make_packet(0, pts, false, 4);
            result = il.push(pkt, &stream, 1, 0, identity_timecode);
        }

        let timecodes: Vec<i64> = result.iter().map(|b| b.abs_timecode).collect();
        assert_eq!(timecodes, vec![100, 200, 300]);
    }

    #[test]
    fn test_interleaver_parallel_matches_sequential() {
        // Run the same packets through two interleavers — one forced sequential,
        // one parallel — and verify the payloads are identical.

        let n = 20usize;
        let stream = make_stream(0);

        let mut seq_il = ParallelInterleaver::with_config(
            InterleaverConfig::new()
                .with_max_pending(n + 1)
                .with_parallel_threshold(n + 1), // always sequential
        );
        let mut par_il = ParallelInterleaver::with_config(
            InterleaverConfig::new()
                .with_max_pending(n + 1)
                .with_parallel_threshold(1), // always parallel
        );

        let packets: Vec<Packet> = (0..n)
            .map(|i| make_packet(0, i as i64 * 50, i == 0, 8))
            .collect();

        for pkt in &packets {
            seq_il.push(pkt.clone(), &stream, 1, 0, identity_timecode);
            par_il.push(pkt.clone(), &stream, 1, 0, identity_timecode);
        }

        let seq_blocks = seq_il.flush();
        let par_blocks = par_il.flush();

        assert_eq!(seq_blocks.len(), par_blocks.len());
        for (s, p) in seq_blocks.iter().zip(par_blocks.iter()) {
            assert_eq!(s.payload, p.payload, "payloads differ at timecode {}", s.abs_timecode);
            assert_eq!(s.is_keyframe, p.is_keyframe);
            assert_eq!(s.rel_timecode, p.rel_timecode);
        }
    }

    #[test]
    fn test_interleaver_multi_track() {
        let config = InterleaverConfig::new()
            .with_max_pending(4)
            .with_parallel_threshold(2);
        let mut il = ParallelInterleaver::with_config(config);

        let video = make_stream(0);
        let audio = make_stream(1);

        // Interleave video (track 1) and audio (track 2)
        il.push(make_packet(0, 0, true, 100), &video, 1, 0, identity_timecode);
        il.push(make_packet(1, 10, false, 20), &audio, 2, 0, identity_timecode);
        il.push(make_packet(0, 33, false, 90), &video, 1, 0, identity_timecode);
        let blocks = il.push(make_packet(1, 43, false, 20), &audio, 2, 0, identity_timecode);

        assert_eq!(blocks.len(), 4);
        // First block should be at timecode 0
        assert_eq!(blocks[0].abs_timecode, 0);
    }

    #[test]
    fn test_config_defaults() {
        let c = InterleaverConfig::default();
        assert_eq!(c.max_pending, 32);
        assert_eq!(c.parallel_threshold, 4);
    }

    #[test]
    fn test_config_builder() {
        let c = InterleaverConfig::new()
            .with_max_pending(16)
            .with_parallel_threshold(8);
        assert_eq!(c.max_pending, 16);
        assert_eq!(c.parallel_threshold, 8);
    }
}
