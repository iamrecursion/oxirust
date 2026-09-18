//! Codec packet and frame building utilities.
//!
//! This module provides:
//!
//! - [`CodecPacket`] — a rich packet type carrying timestamps, flags, and payload.
//! - [`PacketFlags`] — per-packet Boolean flags (keyframe, corrupt, discard).
//! - [`PacketBuilder`] — a stateful helper that produces correctly-timestamped
//!   [`CodecPacket`]s for video and audio streams.
//! - [`PacketReorderer`] — a bounded priority queue that converts DTS-ordered
//!   packets (from the encoder) into PTS order (for the muxer / consumer).
//!
//! # Timestamp arithmetic
//!
//! All timestamps are expressed in **time-base units**.  A time base of
//! `(1, 90000)` means each unit represents 1/90000 of a second.  The helpers
//! [`CodecPacket::pts_secs`] and [`CodecPacket::dts_secs`] convert to seconds.
//! [`CodecPacket::rebase`] rescales all timestamps to a different time base.
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::packet_builder::PacketBuilder;
//!
//! // Build video packets at 30 fps with 90 kHz time base.
//! let mut builder = PacketBuilder::new(0, (1, 90_000), 30.0);
//! let pkt = builder.build_video_frame(vec![0xAB; 1024], true);
//! assert!(pkt.flags.keyframe);
//! assert_eq!(pkt.stream_index, 0);
//! ```

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

// ──────────────────────────────────────────────
// PacketFlags
// ──────────────────────────────────────────────

/// Per-packet boolean flags.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PacketFlags {
    /// The packet contains a keyframe (random access point).
    pub keyframe: bool,
    /// The packet data may be corrupt or partially lost.
    pub corrupt: bool,
    /// The packet should be decoded but not displayed.
    pub discard: bool,
}

// ──────────────────────────────────────────────
// CodecPacket
// ──────────────────────────────────────────────

/// A single codec-level packet (compressed frame data + timestamps).
///
/// Timestamps are stored as unsigned integers in time-base units.  The
/// time base is carried in the packet itself so that consumers do not need
/// out-of-band information to interpret the timestamps.
#[derive(Debug, Clone)]
pub struct CodecPacket {
    /// Presentation timestamp — when this frame should be displayed.
    pub pts: u64,
    /// Decode timestamp — when this frame must be decoded.
    pub dts: u64,
    /// Frame duration in time-base units.
    pub duration: u32,
    /// Time base as `(numerator, denominator)`.
    /// Seconds = `pts * numerator / denominator`.
    pub time_base: (u32, u32),
    /// Compressed frame payload.
    pub data: Vec<u8>,
    /// Boolean flags.
    pub flags: PacketFlags,
    /// Index of the stream this packet belongs to.
    pub stream_index: u32,
}

impl CodecPacket {
    /// PTS in seconds.
    #[must_use]
    pub fn pts_secs(&self) -> f64 {
        let (num, den) = self.time_base;
        if den == 0 {
            return 0.0;
        }
        self.pts as f64 * num as f64 / den as f64
    }

    /// DTS in seconds.
    #[must_use]
    pub fn dts_secs(&self) -> f64 {
        let (num, den) = self.time_base;
        if den == 0 {
            return 0.0;
        }
        self.dts as f64 * num as f64 / den as f64
    }

    /// Rescale all timestamps to `new_time_base`, returning a new packet.
    ///
    /// The rescaling uses 64-bit integer arithmetic with rounding, matching
    /// the behaviour of `av_rescale_rnd(…, AV_ROUND_NEAR_INF)`.
    #[must_use]
    pub fn rebase(&self, new_time_base: (u32, u32)) -> Self {
        let (old_num, old_den) = self.time_base;
        let (new_num, new_den) = new_time_base;

        // Convert: value_in_new = value_in_old * old_num * new_den / (old_den * new_num)
        // Intermediate values are computed in u128 to prevent overflow.
        let rescale = |v: u64| -> u64 {
            if old_den == 0 || new_num == 0 {
                return v;
            }
            let numerator = v as u128 * old_num as u128 * new_den as u128;
            let denominator = old_den as u128 * new_num as u128;
            if denominator == 0 {
                return v;
            }
            ((numerator + denominator / 2) / denominator) as u64
        };

        let dur_rescale = |v: u32| -> u32 {
            if old_den == 0 || new_num == 0 {
                return v;
            }
            let numerator = v as u128 * old_num as u128 * new_den as u128;
            let denominator = old_den as u128 * new_num as u128;
            if denominator == 0 {
                return v;
            }
            ((numerator + denominator / 2) / denominator).min(u32::MAX as u128) as u32
        };

        Self {
            pts: rescale(self.pts),
            dts: rescale(self.dts),
            duration: dur_rescale(self.duration),
            time_base: new_time_base,
            data: self.data.clone(),
            flags: self.flags.clone(),
            stream_index: self.stream_index,
        }
    }
}

// ──────────────────────────────────────────────
// PacketBuilder
// ──────────────────────────────────────────────

/// A stateful helper for building correctly-timestamped [`CodecPacket`]s.
///
/// Create one builder per stream.  Call [`build_video_frame`] for each video
/// frame or [`build_audio_frame`] for each audio frame; the builder tracks
/// the running PTS/DTS automatically.
///
/// [`build_video_frame`]: PacketBuilder::build_video_frame
/// [`build_audio_frame`]: PacketBuilder::build_audio_frame
pub struct PacketBuilder {
    /// Stream index embedded in every produced packet.
    stream_index: u32,
    /// Time base embedded in every produced packet.
    time_base: (u32, u32),
    /// Next PTS to assign (incremented by `frame_duration` after each call).
    pts_counter: u64,
    /// Next DTS to assign.
    dts_counter: u64,
    /// Duration of one video frame in time-base units.
    frame_duration: u32,
}

impl PacketBuilder {
    /// Create a new builder.
    ///
    /// - `stream_index`: stream index embedded into every packet.
    /// - `time_base`: `(numerator, denominator)` of the stream time base.
    /// - `fps`: frame rate; used to compute `frame_duration`.
    ///
    /// `frame_duration` is computed as
    /// `round(time_base.denominator / (fps * time_base.numerator))`,
    /// clamped to at least 1.
    #[must_use]
    pub fn new(stream_index: u32, time_base: (u32, u32), fps: f32) -> Self {
        let (num, den) = time_base;
        let frame_duration = if num == 0 || fps <= 0.0 {
            1
        } else {
            ((den as f64 / (fps as f64 * num as f64)).round() as u32).max(1)
        };

        Self {
            stream_index,
            time_base,
            pts_counter: 0,
            dts_counter: 0,
            frame_duration,
        }
    }

    /// Build a video frame packet and advance the timestamp counters.
    ///
    /// The PTS and DTS of the returned packet reflect the state **before** the
    /// counters are advanced, so the first packet always has PTS/DTS = 0.
    pub fn build_video_frame(&mut self, data: Vec<u8>, keyframe: bool) -> CodecPacket {
        let pkt = CodecPacket {
            pts: self.pts_counter,
            dts: self.dts_counter,
            duration: self.frame_duration,
            time_base: self.time_base,
            data,
            flags: PacketFlags {
                keyframe,
                corrupt: false,
                discard: false,
            },
            stream_index: self.stream_index,
        };

        self.pts_counter = self.pts_counter.saturating_add(self.frame_duration as u64);
        self.dts_counter = self.dts_counter.saturating_add(self.frame_duration as u64);
        pkt
    }

    /// Build an audio frame packet.
    ///
    /// `samples` is the number of PCM samples in the frame.  The duration is
    /// derived from `samples * time_base.numerator / sample_rate`, but since
    /// `PacketBuilder` is not audio-sample-rate aware, the caller should pass
    /// the actual per-frame sample count and the method uses `frame_duration`
    /// as a fallback when `samples == 0`.
    ///
    /// For audio the `keyframe` flag is always `false` (all audio frames are
    /// independently decodable).
    pub fn build_audio_frame(&mut self, data: Vec<u8>, samples: u32) -> CodecPacket {
        let duration = if samples > 0 {
            samples
        } else {
            self.frame_duration
        };

        let pkt = CodecPacket {
            pts: self.pts_counter,
            dts: self.dts_counter,
            duration,
            time_base: self.time_base,
            data,
            flags: PacketFlags {
                keyframe: false,
                corrupt: false,
                discard: false,
            },
            stream_index: self.stream_index,
        };

        self.pts_counter = self.pts_counter.saturating_add(duration as u64);
        self.dts_counter = self.dts_counter.saturating_add(duration as u64);
        pkt
    }

    /// Return the current PTS counter (PTS that will be assigned to the next packet).
    #[must_use]
    pub fn next_pts(&self) -> u64 {
        self.pts_counter
    }

    /// Return the current frame duration in time-base units.
    #[must_use]
    pub fn frame_duration(&self) -> u32 {
        self.frame_duration
    }
}

// ──────────────────────────────────────────────
// PacketReorderer
// ──────────────────────────────────────────────

/// A wrapper that makes [`CodecPacket`] comparable by PTS for the heap.
///
/// The heap orders by `Reverse((pts, dts))` so that the packet with the
/// smallest PTS is always at the top.
#[derive(Debug)]
struct HeapEntry(u64, u64, CodecPacket); // (pts, dts, packet)

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Order by PTS ascending, break ties by DTS ascending.
        (self.0, self.1).cmp(&(other.0, other.1))
    }
}

/// Reorders DTS-ordered packets (as produced by encoders with B-frames) into
/// PTS order (required by muxers and decoders operating on display order).
///
/// Internally uses a min-heap keyed on PTS.  A packet is considered *ready*
/// once the heap has accumulated at least `max_buffer` entries, which bounds
/// the maximum PTS reordering delay.
///
/// # Flushing
///
/// Call [`drain`] at end-of-stream to retrieve all remaining packets in PTS
/// order.
///
/// [`drain`]: PacketReorderer::drain
pub struct PacketReorderer {
    /// Min-heap of `Reverse(HeapEntry)` so the smallest PTS surfaces first.
    buffer: BinaryHeap<Reverse<HeapEntry>>,
    /// Maximum packets buffered before [`pop_ready`] will return a packet.
    ///
    /// [`pop_ready`]: PacketReorderer::pop_ready
    max_buffer: usize,
}

impl PacketReorderer {
    /// Create a new reorderer.
    ///
    /// `max_buffer` controls the maximum reorder window.  A value of 4–8 is
    /// appropriate for streams with up to 3 consecutive B-frames.
    #[must_use]
    pub fn new(max_buffer: usize) -> Self {
        Self {
            buffer: BinaryHeap::with_capacity(max_buffer + 1),
            max_buffer: max_buffer.max(1),
        }
    }

    /// Push a packet into the reorder buffer.
    pub fn push(&mut self, pkt: CodecPacket) {
        let entry = HeapEntry(pkt.pts, pkt.dts, pkt);
        self.buffer.push(Reverse(entry));
    }

    /// Pop the packet with the lowest PTS if the buffer is full enough to
    /// guarantee it is the next in display order.
    ///
    /// Returns `None` if the buffer is smaller than `max_buffer`.
    pub fn pop_ready(&mut self) -> Option<CodecPacket> {
        if self.buffer.len() >= self.max_buffer {
            self.buffer.pop().map(|Reverse(HeapEntry(_, _, pkt))| pkt)
        } else {
            None
        }
    }

    /// Drain all remaining packets from the buffer, ordered by PTS ascending.
    ///
    /// The buffer is empty after this call.
    pub fn drain(&mut self) -> Vec<CodecPacket> {
        let mut out = Vec::with_capacity(self.buffer.len());
        while let Some(Reverse(HeapEntry(_, _, pkt))) = self.buffer.pop() {
            out.push(pkt);
        }
        out
    }

    /// Number of packets currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the buffer contains no packets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. PacketBuilder: first packet has PTS/DTS = 0 ──────────────────────

    #[test]
    fn builder_first_pts_zero() {
        let mut b = PacketBuilder::new(0, (1, 90_000), 30.0);
        let p = b.build_video_frame(vec![0u8; 10], true);
        assert_eq!(p.pts, 0, "first packet PTS must be 0");
        assert_eq!(p.dts, 0, "first packet DTS must be 0");
    }

    // ── 2. PacketBuilder: consecutive frames advance PTS by frame_duration ───

    #[test]
    fn builder_pts_advances() {
        let mut b = PacketBuilder::new(0, (1, 90_000), 30.0);
        let dur = b.frame_duration();
        let p0 = b.build_video_frame(vec![], true);
        let p1 = b.build_video_frame(vec![], false);
        assert_eq!(
            p1.pts - p0.pts,
            dur as u64,
            "PTS must advance by frame_duration"
        );
    }

    // ── 3. PacketBuilder: keyframe flag is propagated ────────────────────────

    #[test]
    fn builder_keyframe_flag() {
        let mut b = PacketBuilder::new(1, (1, 90_000), 25.0);
        let key = b.build_video_frame(vec![], true);
        let non_key = b.build_video_frame(vec![], false);
        assert!(key.flags.keyframe);
        assert!(!non_key.flags.keyframe);
    }

    // ── 4. PacketBuilder: stream_index is embedded ───────────────────────────

    #[test]
    fn builder_stream_index() {
        let mut b = PacketBuilder::new(42, (1, 44_100), 25.0);
        let p = b.build_audio_frame(vec![0u8; 4], 1024);
        assert_eq!(p.stream_index, 42);
    }

    // ── 5. PacketBuilder: audio frame keyframe is always false ───────────────

    #[test]
    fn builder_audio_no_keyframe() {
        let mut b = PacketBuilder::new(1, (1, 44_100), 0.0);
        let p = b.build_audio_frame(vec![], 1024);
        assert!(!p.flags.keyframe);
    }

    // ── 6. PacketBuilder: audio duration equals sample count ─────────────────

    #[test]
    fn builder_audio_duration_from_samples() {
        let mut b = PacketBuilder::new(1, (1, 48_000), 25.0);
        let p = b.build_audio_frame(vec![], 960);
        assert_eq!(p.duration, 960, "audio duration must equal sample count");
    }

    // ── 7. CodecPacket::pts_secs: correct conversion ─────────────────────────

    #[test]
    fn pts_secs_conversion() {
        let pkt = CodecPacket {
            pts: 90_000,
            dts: 90_000,
            duration: 3000,
            time_base: (1, 90_000),
            data: vec![],
            flags: PacketFlags::default(),
            stream_index: 0,
        };
        let secs = pkt.pts_secs();
        assert!(
            (secs - 1.0).abs() < 1e-9,
            "pts_secs should be 1.0, got {secs}"
        );
    }

    // ── 8. CodecPacket::dts_secs: correct conversion ─────────────────────────

    #[test]
    fn dts_secs_conversion() {
        let pkt = CodecPacket {
            pts: 45_000,
            dts: 45_000,
            duration: 3000,
            time_base: (1, 90_000),
            data: vec![],
            flags: PacketFlags::default(),
            stream_index: 0,
        };
        assert!((pkt.dts_secs() - 0.5).abs() < 1e-9);
    }

    // ── 9. CodecPacket::rebase: 90kHz → 1/1000 ──────────────────────────────

    #[test]
    fn rebase_90k_to_1000() {
        let pkt = CodecPacket {
            pts: 90_000,
            dts: 90_000,
            duration: 3_000,
            time_base: (1, 90_000),
            data: vec![],
            flags: PacketFlags::default(),
            stream_index: 0,
        };
        let rebased = pkt.rebase((1, 1_000));
        assert_eq!(
            rebased.pts, 1_000,
            "90000 ticks @ 1/90000 = 1000 ticks @ 1/1000"
        );
        assert_eq!(rebased.duration, 33, "3000/90000 * 1000 ≈ 33 ms");
    }

    // ── 10. PacketReorderer: empty buffer returns None ───────────────────────

    #[test]
    fn reorderer_empty_returns_none() {
        let mut r = PacketReorderer::new(4);
        assert!(r.pop_ready().is_none());
    }

    // ── 11. PacketReorderer: packets released in PTS order ──────────────────

    #[test]
    fn reorderer_pts_order() {
        let mut r = PacketReorderer::new(3);

        // Push 4 packets with scrambled PTS order (simulating B-frames).
        for (pts, dts) in [(0, 0), (3, 1), (1, 2), (2, 3)] {
            let pkt = CodecPacket {
                pts,
                dts,
                duration: 1,
                time_base: (1, 90_000),
                data: vec![],
                flags: PacketFlags::default(),
                stream_index: 0,
            };
            r.push(pkt);
        }

        // With max_buffer=3 we can pop once buffer >= 3.
        let mut pts_order = Vec::new();
        while let Some(p) = r.pop_ready() {
            pts_order.push(p.pts);
        }
        let remaining = r.drain();
        for p in remaining {
            pts_order.push(p.pts);
        }

        let mut sorted = pts_order.clone();
        sorted.sort_unstable();
        assert_eq!(
            pts_order, sorted,
            "packets must emerge in PTS ascending order"
        );
    }

    // ── 12. PacketReorderer::drain: returns all packets ─────────────────────

    #[test]
    fn reorderer_drain_all() {
        let mut r = PacketReorderer::new(8);
        for i in 0..5_u64 {
            let pkt = CodecPacket {
                pts: 4 - i, // reverse order
                dts: i,
                duration: 1,
                time_base: (1, 25),
                data: vec![],
                flags: PacketFlags::default(),
                stream_index: 0,
            };
            r.push(pkt);
        }
        let drained = r.drain();
        assert_eq!(drained.len(), 5, "drain must return all 5 packets");
        // Check ascending PTS order after drain.
        let pts: Vec<u64> = drained.iter().map(|p| p.pts).collect();
        let mut sorted = pts.clone();
        sorted.sort_unstable();
        assert_eq!(pts, sorted, "drained packets must be in PTS order");
    }
}
