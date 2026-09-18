//! Seeking infrastructure for demuxers.
//!
//! This module provides types and utilities for seeking in media containers,
//! including keyframe-based seeking, sample-accurate seeking, and a seek index
//! for fast random access.
//!
//! # Sample-Accurate Seeking
//!
//! Sample-accurate seeking goes beyond simple keyframe-based seeking by:
//! 1. Finding the nearest keyframe before the target
//! 2. Tracking samples between that keyframe and the target
//! 3. Providing a `SeekPlan` that tells the decoder which samples to decode
//!    and which to discard
//!
//! ```ignore
//! let index = SeekIndex::new(90000); // 90kHz timescale
//! // ... populate with sample entries ...
//! let plan = index.plan_seek(target_pts, SeekAccuracy::SampleAccurate)?;
//! // plan.decode_from_pts: start decoding here (keyframe)
//! // plan.discard_count: number of frames to decode but discard
//! // plan.target_pts: the actual target presentation time
//! ```

use bitflags::bitflags;
use std::cmp::Ordering;
use std::collections::HashMap;

bitflags! {
    /// Flags controlling seek behavior.
    ///
    /// These flags allow fine-grained control over how a seek operation
    /// is performed and what position is targeted.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct SeekFlags: u32 {
        /// Seek backward (to position <= target).
        ///
        /// Without this flag, seeks go forward (to position >= target).
        /// This is useful for finding the keyframe before a target position.
        const BACKWARD = 0x0001;

        /// Allow seeking to any frame, not just keyframes.
        ///
        /// By default, seeks target keyframes for clean decoding.
        /// Setting this flag allows seeking to any position, which may
        /// require decoding from the previous keyframe.
        const ANY = 0x0002;

        /// Seek to the nearest keyframe.
        ///
        /// This is the default behavior and ensures the seek position
        /// can be decoded immediately without reference frames.
        const KEYFRAME = 0x0004;

        /// Seek by bytes rather than time.
        ///
        /// When set, the seek target is interpreted as a byte offset
        /// in the file rather than a timestamp.
        const BYTE = 0x0008;

        /// Seek to exact position (frame-accurate).
        ///
        /// Attempts to seek to the exact target timestamp, which may
        /// require additional parsing and decoding.
        const FRAME_ACCURATE = 0x0010;
    }
}

/// Target for a seek operation.
///
/// Specifies where to seek and which stream to use as reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeekTarget {
    /// Target timestamp in seconds, or byte offset if `SeekFlags::BYTE` is set.
    pub position: f64,

    /// Stream index to use for seeking, or `None` for the default stream.
    ///
    /// The default stream is typically the first video stream, or the
    /// first audio stream if there are no video streams.
    pub stream_index: Option<usize>,

    /// Seek flags controlling behavior.
    pub flags: SeekFlags,
}

impl SeekTarget {
    /// Creates a new seek target to a timestamp in seconds.
    #[must_use]
    pub const fn time(position: f64) -> Self {
        Self {
            position,
            stream_index: None,
            flags: SeekFlags::KEYFRAME,
        }
    }

    /// Creates a new seek target to a byte offset.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn byte(offset: u64) -> Self {
        Self {
            position: offset as f64,
            stream_index: None,
            flags: SeekFlags::BYTE,
        }
    }

    /// Creates a sample-accurate seek target.
    ///
    /// This will seek to the exact position, decoding from the
    /// preceding keyframe and discarding intermediate frames.
    #[must_use]
    pub const fn sample_accurate(position: f64) -> Self {
        Self {
            position,
            stream_index: None,
            flags: SeekFlags::from_bits_truncate(
                SeekFlags::FRAME_ACCURATE.bits() | SeekFlags::BACKWARD.bits(),
            ),
        }
    }

    /// Sets the stream index for this seek target.
    #[must_use]
    pub const fn with_stream(mut self, stream_index: usize) -> Self {
        self.stream_index = Some(stream_index);
        self
    }

    /// Sets the seek flags for this seek target.
    #[must_use]
    pub const fn with_flags(mut self, flags: SeekFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Adds additional flags to this seek target.
    #[must_use]
    pub const fn add_flags(mut self, flags: SeekFlags) -> Self {
        self.flags = SeekFlags::from_bits_truncate(self.flags.bits() | flags.bits());
        self
    }

    /// Returns true if this is a backward seek.
    #[must_use]
    pub const fn is_backward(&self) -> bool {
        self.flags.contains(SeekFlags::BACKWARD)
    }

    /// Returns true if this allows seeking to any frame.
    #[must_use]
    pub const fn is_any(&self) -> bool {
        self.flags.contains(SeekFlags::ANY)
    }

    /// Returns true if this seeks to a keyframe.
    #[must_use]
    pub const fn is_keyframe(&self) -> bool {
        self.flags.contains(SeekFlags::KEYFRAME)
    }

    /// Returns true if this is a byte-based seek.
    #[must_use]
    pub const fn is_byte(&self) -> bool {
        self.flags.contains(SeekFlags::BYTE)
    }

    /// Returns true if this is a frame-accurate seek.
    #[must_use]
    pub const fn is_frame_accurate(&self) -> bool {
        self.flags.contains(SeekFlags::FRAME_ACCURATE)
    }
}

// ─── Seek Accuracy ──────────────────────────────────────────────────────────

/// Desired accuracy level for seeking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekAccuracy {
    /// Seek to the nearest keyframe (fastest, least accurate).
    Keyframe,
    /// Seek to the exact sample/frame (requires decoding from prior keyframe).
    SampleAccurate,
    /// Seek to within a specified tolerance in timescale ticks.
    WithinTolerance(u64),
}

// ─── SeekMode ───────────────────────────────────────────────────────────────

/// High-level seek mode selecting the accuracy/cost tradeoff.
///
/// `KeyframeApproximate` snaps to the nearest keyframe (O(log n), zero preroll
/// decoding).  `SampleAccurate` decodes from the keyframe up to the target,
/// paying up to `max_preroll_frames` extra decode iterations for exact
/// positioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeekMode {
    /// Seek to the nearest keyframe at or before the target (fast, inexact).
    #[default]
    KeyframeApproximate,
    /// Decode from the keyframe to the exact target sample (slower, exact).
    SampleAccurate,
}

// ─── Seek Index Entry ───────────────────────────────────────────────────────

/// An entry in the seek index representing one sample/frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeekIndexEntry {
    /// Presentation timestamp in timescale ticks.
    pub pts: i64,
    /// Decode timestamp in timescale ticks.
    pub dts: i64,
    /// Byte offset in the container file.
    pub file_offset: u64,
    /// Sample size in bytes.
    pub size: u32,
    /// Sample duration in timescale ticks.
    pub duration: u32,
    /// Whether this is a keyframe (sync sample).
    pub is_keyframe: bool,
    /// Sample number (0-based).
    pub sample_number: u32,
}

impl SeekIndexEntry {
    /// Creates a new keyframe entry.
    #[must_use]
    pub const fn keyframe(
        pts: i64,
        dts: i64,
        file_offset: u64,
        size: u32,
        duration: u32,
        sample_number: u32,
    ) -> Self {
        Self {
            pts,
            dts,
            file_offset,
            size,
            duration,
            is_keyframe: true,
            sample_number,
        }
    }

    /// Creates a new non-keyframe entry.
    #[must_use]
    pub const fn non_keyframe(
        pts: i64,
        dts: i64,
        file_offset: u64,
        size: u32,
        duration: u32,
        sample_number: u32,
    ) -> Self {
        Self {
            pts,
            dts,
            file_offset,
            size,
            duration,
            is_keyframe: false,
            sample_number,
        }
    }

    /// Returns the end PTS (pts + duration).
    #[must_use]
    pub const fn end_pts(&self) -> i64 {
        self.pts + self.duration as i64
    }
}

// ─── Seek Plan ──────────────────────────────────────────────────────────────

/// A plan for executing a sample-accurate seek.
///
/// Contains information about where to start decoding (keyframe),
/// how many samples to decode-and-discard, and the final target sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeekPlan {
    /// The keyframe entry to seek to in the container (start decoding here).
    pub keyframe_entry: SeekIndexEntry,
    /// Number of samples between the keyframe and the target to decode
    /// but discard (not presented).
    pub discard_count: u32,
    /// The target sample entry.
    pub target_entry: SeekIndexEntry,
    /// File offset to seek to.
    pub file_offset: u64,
    /// The target PTS that was requested.
    pub requested_pts: i64,
    /// Whether the seek is exact (target PTS matches a sample boundary).
    pub is_exact: bool,
}

// ─── Seek Index ─────────────────────────────────────────────────────────────

/// Index of sample positions for fast seeking.
///
/// Maintains a sorted list of sample entries that enables both keyframe-based
/// and sample-accurate seeking. Entries are sorted by DTS for efficient
/// binary search.
#[derive(Debug, Clone)]
pub struct SeekIndex {
    /// Timescale (ticks per second) for interpreting timestamps.
    timescale: u32,
    /// All sample entries sorted by DTS.
    entries: Vec<SeekIndexEntry>,
    /// Indices of keyframe entries within `entries` (for fast keyframe lookup).
    keyframe_indices: Vec<usize>,
}

impl SeekIndex {
    /// Creates a new empty seek index.
    #[must_use]
    pub fn new(timescale: u32) -> Self {
        Self {
            timescale,
            entries: Vec::new(),
            keyframe_indices: Vec::new(),
        }
    }

    /// Creates a seek index with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(timescale: u32, capacity: usize) -> Self {
        Self {
            timescale,
            entries: Vec::with_capacity(capacity),
            keyframe_indices: Vec::new(),
        }
    }

    /// Returns the timescale.
    #[must_use]
    pub const fn timescale(&self) -> u32 {
        self.timescale
    }

    /// Returns the number of entries in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of keyframes in the index.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframe_indices.len()
    }

    /// Returns all entries.
    #[must_use]
    pub fn entries(&self) -> &[SeekIndexEntry] {
        &self.entries
    }

    /// Adds a sample entry to the index.
    ///
    /// Entries should be added in DTS order for optimal performance.
    /// If entries are added out of order, call [`sort`](SeekIndex::sort)
    /// before seeking.
    pub fn add_entry(&mut self, entry: SeekIndexEntry) {
        let idx = self.entries.len();
        if entry.is_keyframe {
            self.keyframe_indices.push(idx);
        }
        self.entries.push(entry);
    }

    /// Sorts entries by DTS and rebuilds the keyframe index.
    pub fn sort(&mut self) {
        self.entries.sort_by_key(|e| e.dts);
        self.keyframe_indices.clear();
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.is_keyframe {
                self.keyframe_indices.push(i);
            }
        }
    }

    /// Finalizes the index after all entries have been added.
    ///
    /// Equivalent to [`SeekIndex::sort`]: sorts all entries by DTS and
    /// rebuilds the keyframe lookup table.  Call this once after all calls
    /// to [`SeekIndex::add_entry`] are complete.
    pub fn finalize(&mut self) {
        self.sort();
    }

    /// Converts a time in seconds to ticks in this index's timescale.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn seconds_to_ticks(&self, seconds: f64) -> i64 {
        (seconds * f64::from(self.timescale)) as i64
    }

    /// Converts ticks to seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn ticks_to_seconds(&self, ticks: i64) -> f64 {
        if self.timescale == 0 {
            return 0.0;
        }
        ticks as f64 / f64::from(self.timescale)
    }

    /// Finds the nearest keyframe at or before the given PTS.
    ///
    /// Returns `None` if the index is empty or has no keyframes.
    #[must_use]
    pub fn find_keyframe_before(&self, target_pts: i64) -> Option<&SeekIndexEntry> {
        if self.keyframe_indices.is_empty() {
            return None;
        }

        let mut best: Option<usize> = None;
        let mut lo = 0usize;
        let mut hi = self.keyframe_indices.len();

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let kf_idx = self.keyframe_indices[mid];
            let kf = &self.entries[kf_idx];

            match kf.pts.cmp(&target_pts) {
                Ordering::Less | Ordering::Equal => {
                    best = Some(kf_idx);
                    lo = mid + 1;
                }
                Ordering::Greater => {
                    hi = mid;
                }
            }
        }

        best.map(|idx| &self.entries[idx])
    }

    /// Finds the nearest keyframe at or after the given PTS.
    ///
    /// Returns `None` if no keyframe exists at or after the target.
    #[must_use]
    pub fn find_keyframe_after(&self, target_pts: i64) -> Option<&SeekIndexEntry> {
        if self.keyframe_indices.is_empty() {
            return None;
        }

        let mut lo = 0usize;
        let mut hi = self.keyframe_indices.len();

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let kf_idx = self.keyframe_indices[mid];
            let kf = &self.entries[kf_idx];

            if kf.pts < target_pts {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if lo < self.keyframe_indices.len() {
            Some(&self.entries[self.keyframe_indices[lo]])
        } else {
            None
        }
    }

    /// Finds the nearest keyframe (before or after) to the given PTS.
    ///
    /// Returns whichever keyframe is closer in PTS distance.
    #[must_use]
    pub fn find_nearest_keyframe(&self, target_pts: i64) -> Option<&SeekIndexEntry> {
        let before = self.find_keyframe_before(target_pts);
        let after = self.find_keyframe_after(target_pts);

        match (before, after) {
            (None, None) => None,
            (Some(b), None) => Some(b),
            (None, Some(a)) => Some(a),
            (Some(b), Some(a)) => {
                let dist_before = (target_pts - b.pts).unsigned_abs();
                let dist_after = (a.pts - target_pts).unsigned_abs();
                if dist_before <= dist_after {
                    Some(b)
                } else {
                    Some(a)
                }
            }
        }
    }

    /// Finds the exact sample entry whose PTS range contains the target.
    ///
    /// Returns `None` if no sample covers the target PTS.
    #[must_use]
    pub fn find_sample_at(&self, target_pts: i64) -> Option<&SeekIndexEntry> {
        let result = self.entries.binary_search_by(|entry| {
            if entry.pts > target_pts {
                Ordering::Greater
            } else if entry.end_pts() <= target_pts {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        });

        match result {
            Ok(idx) => Some(&self.entries[idx]),
            Err(_) => {
                let mut best = None;
                for entry in &self.entries {
                    if entry.pts <= target_pts {
                        best = Some(entry);
                    } else {
                        break;
                    }
                }
                best
            }
        }
    }

    /// Plans a sample-accurate seek to the given PTS.
    ///
    /// Returns a `SeekPlan` describing how to execute the seek:
    /// - Which keyframe to seek to in the container
    /// - How many samples to decode and discard
    /// - The target sample
    ///
    /// Returns `None` if the index is empty or the target is out of range.
    #[must_use]
    pub fn plan_seek(&self, target_pts: i64, accuracy: SeekAccuracy) -> Option<SeekPlan> {
        if self.entries.is_empty() || self.keyframe_indices.is_empty() {
            return None;
        }

        match accuracy {
            SeekAccuracy::Keyframe => {
                let kf = self.find_keyframe_before(target_pts)?;
                Some(SeekPlan {
                    keyframe_entry: *kf,
                    discard_count: 0,
                    target_entry: *kf,
                    file_offset: kf.file_offset,
                    requested_pts: target_pts,
                    is_exact: kf.pts == target_pts,
                })
            }
            SeekAccuracy::SampleAccurate => self.plan_sample_accurate_seek(target_pts),
            SeekAccuracy::WithinTolerance(tolerance) => {
                if let Some(plan) = self.plan_sample_accurate_seek(target_pts) {
                    let distance = (plan.target_entry.pts - target_pts).unsigned_abs();
                    if distance <= tolerance {
                        return Some(plan);
                    }
                }
                let kf = self.find_nearest_keyframe(target_pts)?;
                let distance = (kf.pts - target_pts).unsigned_abs();
                if distance <= tolerance {
                    Some(SeekPlan {
                        keyframe_entry: *kf,
                        discard_count: 0,
                        target_entry: *kf,
                        file_offset: kf.file_offset,
                        requested_pts: target_pts,
                        is_exact: kf.pts == target_pts,
                    })
                } else {
                    None
                }
            }
        }
    }

    fn plan_sample_accurate_seek(&self, target_pts: i64) -> Option<SeekPlan> {
        let kf = self.find_keyframe_before(target_pts)?;
        let kf_copy = *kf;

        let target_sample = self.find_sample_at(target_pts);
        let target = match target_sample {
            Some(s) => *s,
            None => *self.entries.last()?,
        };

        let mut discard_count: u32 = 0;
        for entry in &self.entries {
            if entry.dts > kf_copy.dts && entry.dts < target.dts {
                discard_count += 1;
            }
        }

        Some(SeekPlan {
            keyframe_entry: kf_copy,
            discard_count,
            target_entry: target,
            file_offset: kf_copy.file_offset,
            requested_pts: target_pts,
            is_exact: target.pts <= target_pts && target_pts < target.end_pts(),
        })
    }

    /// Returns the duration of the indexed content in timescale ticks.
    #[must_use]
    pub fn duration_ticks(&self) -> i64 {
        self.entries.last().map_or(0, |e| e.pts + e.duration as i64)
    }

    /// Returns the duration of the indexed content in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.ticks_to_seconds(self.duration_ticks())
    }

    /// Returns the average keyframe interval in timescale ticks.
    #[must_use]
    pub fn average_keyframe_interval(&self) -> Option<f64> {
        if self.keyframe_indices.len() < 2 {
            return None;
        }

        let mut total_interval: i64 = 0;
        for i in 1..self.keyframe_indices.len() {
            let prev = &self.entries[self.keyframe_indices[i - 1]];
            let curr = &self.entries[self.keyframe_indices[i]];
            total_interval += curr.pts - prev.pts;
        }

        #[allow(clippy::cast_precision_loss)]
        let avg = total_interval as f64 / (self.keyframe_indices.len() - 1) as f64;
        Some(avg)
    }
}

/// Type alias for [`SeekIndex`] used in pre-roll seeking contexts.
pub type SampleIndex = SeekIndex;

// ─── TrackIndex ─────────────────────────────────────────────────────────────

/// A lightweight index of keyframe positions within a single track.
///
/// Used by [`SampleAccurateSeeker`] to locate the nearest keyframe before a
/// target PTS and compute the number of samples that must be decoded and
/// discarded to reach a sample-accurate position.
#[derive(Debug, Clone)]
pub struct TrackIndex {
    /// The underlying seek index (sorted by DTS).
    pub seek_index: SeekIndex,
    /// Codec delay in samples (e.g. 512 for Opus, 0 for most video codecs).
    /// Added to the `preroll_samples` field of the returned [`SeekResult`].
    pub codec_delay_samples: u32,
}

impl TrackIndex {
    /// Creates a `TrackIndex` from an existing [`SeekIndex`].
    #[must_use]
    pub fn new(seek_index: SeekIndex) -> Self {
        Self {
            seek_index,
            codec_delay_samples: 0,
        }
    }

    /// Creates a `TrackIndex` with an explicit codec delay.
    #[must_use]
    pub fn with_codec_delay(seek_index: SeekIndex, codec_delay_samples: u32) -> Self {
        Self {
            seek_index,
            codec_delay_samples,
        }
    }
}

// ─── SeekResult ─────────────────────────────────────────────────────────────

/// The result of a sample-accurate seek operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeekResult {
    /// The PTS of the keyframe that the decoder must start from.
    pub keyframe_pts: u64,
    /// Byte offset of the keyframe in the container file.
    pub sample_offset: u64,
    /// Number of samples to decode and discard between `keyframe_pts` and the
    /// target PTS, plus any codec delay.
    ///
    /// A value of 0 means the seek landed exactly on a keyframe boundary.
    pub preroll_samples: u32,
}

/// A shared decode-and-skip cursor for sample-accurate seek planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeSkipCursor {
    /// Byte offset of the keyframe/sample where decoding should begin.
    pub byte_offset: u64,
    /// 0-based sample index where decoding should begin.
    pub sample_index: usize,
    /// Number of decoded samples to discard before presentation.
    pub skip_samples: u32,
    /// Requested target presentation timestamp in track timescale units.
    pub target_pts: i64,
}

// ─── ClosedLoopSeekError ─────────────────────────────────────────────────────

/// Error type for [`SampleAccurateSeeker::seek_to_pts`] closure-based seeking.
#[derive(Debug, thiserror::Error)]
pub enum ClosedLoopSeekError {
    /// The seek callback reported an error.
    #[error("seek callback error: {0}")]
    SeekCallbackError(String),
    /// The decode callback reported an error.
    #[error("decode callback error: {0}")]
    DecodeCallbackError(String),
    /// No keyframe was found at or before the target PTS.
    #[error("no keyframe found at or before PTS {0}")]
    NoKeyframe(u64),
    /// Pre-roll decode limit exceeded before reaching the target PTS.
    #[error(
        "preroll limit {limit} exceeded: decoded {decoded} frames without reaching PTS {target}"
    )]
    MaxPrerollExceeded {
        /// The configured frame limit.
        limit: u32,
        /// Frames decoded before giving up.
        decoded: u32,
        /// The target PTS that was never reached.
        target: u64,
    },
    /// The decode stream ended before the target PTS was reached.
    #[error("stream ended before reaching target PTS {0}")]
    StreamEnded(u64),
}

// ─── SampleAccurateSeeker ───────────────────────────────────────────────────

/// Performs sample-accurate seeking on a single media track.
///
/// Wraps a [`TrackIndex`] and provides a high-level `seek_to_sample` method
/// that finds the nearest keyframe at or before a target PTS, returning the
/// exact byte offset and the number of samples that must be decoded and
/// discarded to reach the target position.
pub struct SampleAccurateSeeker {
    /// The primary track index used for single-track seek operations.
    pub track: TrackIndex,
    /// Per-stream sample indices for multi-stream pre-roll seeking.
    streams: HashMap<u32, SampleIndex>,
}

impl SampleAccurateSeeker {
    /// Creates a new multi-stream `SampleAccurateSeeker` with no pre-loaded
    /// streams.
    #[must_use]
    pub fn new() -> Self {
        let empty_index = SeekIndex::new(90_000);
        let empty_track = TrackIndex::new(empty_index);
        Self {
            track: empty_track,
            streams: HashMap::new(),
        }
    }

    /// Creates a `SampleAccurateSeeker` from a single pre-built [`TrackIndex`].
    #[must_use]
    pub fn with_track(track: TrackIndex) -> Self {
        Self {
            track,
            streams: HashMap::new(),
        }
    }

    /// Creates a [`SampleAccurateSeeker`] with a `max_preroll_frames` limit.
    ///
    /// `max_preroll_frames` caps the number of frames decoded during pre-roll.
    /// A value of `0` means only seeks that land exactly on a keyframe succeed.
    #[must_use]
    pub fn with_max_preroll(max_preroll_frames: u32) -> Self {
        let empty_index = SeekIndex::new(90_000);
        let track = TrackIndex::with_codec_delay(empty_index, max_preroll_frames);
        Self {
            track,
            streams: HashMap::new(),
        }
    }

    /// Registers a per-stream `SampleIndex` for multi-stream pre-roll seeking.
    pub fn add_stream(&mut self, stream_id: u32, index: SampleIndex) {
        self.streams.insert(stream_id, index);
    }

    /// Plans a pre-roll seek for `stream_id` to `target_pts`.
    ///
    /// Returns `None` if `stream_id` has not been registered or the index has
    /// no keyframe at or before `target_pts`.
    #[must_use]
    pub fn plan_preroll_seek(
        &self,
        stream_id: u32,
        target_pts: i64,
        max_preroll: Option<u32>,
    ) -> Option<crate::preroll::PreRollSeekPlan> {
        use crate::preroll::{PreRollAction, PreRollSample, PreRollSeekPlan};

        let index = self.streams.get(&stream_id)?;
        let keyframe = index.find_keyframe_before(target_pts)?;

        let all_from_kf: Vec<&SeekIndexEntry> = index
            .entries()
            .iter()
            .filter(|e| e.pts >= keyframe.pts)
            .collect();

        let discard_candidates: Vec<&&SeekIndexEntry> =
            all_from_kf.iter().filter(|e| e.pts < target_pts).collect();
        let present_candidate: Option<&SeekIndexEntry> =
            all_from_kf.iter().find(|e| e.pts >= target_pts).copied();

        let capped_discards: Vec<&SeekIndexEntry> = if let Some(max) = max_preroll {
            let max = max as usize;
            if discard_candidates.len() > max {
                discard_candidates[discard_candidates.len() - max..]
                    .iter()
                    .copied()
                    .copied()
                    .collect()
            } else {
                discard_candidates.iter().copied().copied().collect()
            }
        } else {
            discard_candidates.iter().copied().copied().collect()
        };

        let mut samples: Vec<PreRollSample> = capped_discards
            .iter()
            .map(|e| PreRollSample {
                entry: **e,
                action: PreRollAction::Decode,
            })
            .collect();

        let discard_count = samples.len() as u32;
        let mut present_count: u32 = 0;

        if let Some(entry) = present_candidate {
            samples.push(PreRollSample {
                entry: *entry,
                action: PreRollAction::Present,
            });
            present_count = 1;
        } else if discard_count == 0 {
            return None;
        } else if let Some(last) = samples.last_mut() {
            last.action = PreRollAction::Present;
            present_count = 1;
        }

        let final_discard_count = samples
            .iter()
            .filter(|s| matches!(s.action, PreRollAction::Decode))
            .count() as u32;

        Some(PreRollSeekPlan {
            keyframe: *keyframe,
            target_pts,
            samples,
            discard_count: final_discard_count,
            present_count,
            file_offset: keyframe.file_offset,
        })
    }

    /// Returns the number of samples that must be decoded and discarded
    /// (pre-roll count) to achieve sample-accurate positioning at `target_pts`
    /// in `stream_id`.
    ///
    /// Returns `None` if the stream is not registered or has no suitable
    /// keyframe.
    #[must_use]
    pub fn preroll_count(&self, stream_id: u32, target_pts: i64) -> Option<u32> {
        let plan = self.plan_preroll_seek(stream_id, target_pts, None)?;
        Some(plan.discard_count)
    }

    /// Seeks to the sample-accurate position for `target_pts` within `track`.
    ///
    /// Returns `Some(SeekResult)` if a keyframe is found, or `None` if the
    /// index is empty or no keyframe exists before `target_pts`.
    #[must_use]
    pub fn seek_to_sample(&self, target_pts: u64, track: &TrackIndex) -> Option<SeekResult> {
        let target_i64 = i64::try_from(target_pts).unwrap_or(i64::MAX);

        let plan = track
            .seek_index
            .plan_seek(target_i64, SeekAccuracy::SampleAccurate)?;

        let keyframe_pts = u64::try_from(plan.keyframe_entry.pts.max(0)).unwrap_or(0);
        let sample_offset = plan.keyframe_entry.file_offset;
        let preroll_samples = plan.discard_count.saturating_add(track.codec_delay_samples);

        Some(SeekResult {
            keyframe_pts,
            sample_offset,
            preroll_samples,
        })
    }

    /// Seeks to `target_pts` sample-accurately using caller-provided callbacks.
    ///
    /// This method drives the seek loop externally: it calls `seek_fn` once to
    /// land on the nearest keyframe, then repeatedly calls `decode_fn` until
    /// the current decoded PTS meets or exceeds `target_pts`.
    ///
    /// # Returns
    ///
    /// The number of preroll frames decoded (0 if the seek landed exactly on a
    /// keyframe).
    ///
    /// # Errors
    ///
    /// * [`ClosedLoopSeekError::SeekCallbackError`] — `seek_fn` returned `Err`.
    /// * [`ClosedLoopSeekError::DecodeCallbackError`] — `decode_fn` returned `Err`.
    /// * [`ClosedLoopSeekError::MaxPrerollExceeded`] — more than
    ///   `max_preroll_frames` frames were decoded without reaching `target_pts`.
    /// * [`ClosedLoopSeekError::StreamEnded`] — `decode_fn` returned `None`
    ///   before `target_pts` was reached.
    pub fn seek_to_pts<S, D>(
        &self,
        seek_fn: S,
        mut decode_fn: D,
        target_pts: u64,
    ) -> Result<u32, ClosedLoopSeekError>
    where
        S: Fn(u64) -> Result<u64, String>,
        D: FnMut() -> Result<Option<u64>, String>,
    {
        let max_preroll = self.track.codec_delay_samples;

        let keyframe_pts = seek_fn(target_pts).map_err(ClosedLoopSeekError::SeekCallbackError)?;

        if keyframe_pts >= target_pts {
            return Ok(0);
        }

        let mut preroll_count: u32 = 0;
        loop {
            if preroll_count > max_preroll {
                return Err(ClosedLoopSeekError::MaxPrerollExceeded {
                    limit: max_preroll,
                    decoded: preroll_count,
                    target: target_pts,
                });
            }

            let maybe_pts = decode_fn().map_err(ClosedLoopSeekError::DecodeCallbackError)?;

            match maybe_pts {
                None => return Err(ClosedLoopSeekError::StreamEnded(target_pts)),
                Some(pts) if pts >= target_pts => return Ok(preroll_count),
                Some(_) => preroll_count += 1,
            }
        }
    }
}

impl Default for SampleAccurateSeeker {
    fn default() -> Self {
        Self::new()
    }
}

// ─── MultiTrackSeeker ────────────────────────────────────────────────────────

/// A compact index entry describing a single sample within a track.
///
/// Used by [`MultiTrackSeeker`] to build a per-track PTS→byte-offset index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleIndexEntry {
    /// Presentation timestamp in the track's timescale ticks.
    pub pts: i64,
    /// Byte offset of the sample within the container file.
    pub byte_offset: u64,
    /// Whether this sample is a sync (key) sample.
    pub is_sync: bool,
}

impl SampleIndexEntry {
    /// Creates a new keyframe [`SampleIndexEntry`].
    #[must_use]
    pub const fn keyframe(pts: i64, byte_offset: u64) -> Self {
        Self {
            pts,
            byte_offset,
            is_sync: true,
        }
    }

    /// Creates a new non-keyframe [`SampleIndexEntry`].
    #[must_use]
    pub const fn delta(pts: i64, byte_offset: u64) -> Self {
        Self {
            pts,
            byte_offset,
            is_sync: false,
        }
    }
}

/// The result of a [`MultiTrackSeeker::seek_to_pts`] operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtsSeekResult {
    /// Presentation timestamp of the sample that was found (in timescale ticks).
    pub found_pts: i64,
    /// Byte offset of the found sample in the container file.
    pub byte_offset: u64,
    /// 0-based index of the sample within the track's sorted index array.
    pub sample_idx: usize,
}

/// Error type returned by [`MultiTrackSeeker`] operations.
#[derive(Debug, thiserror::Error)]
pub enum MultiTrackSeekerError {
    /// The requested track ID has not been indexed yet.
    #[error("no index for track {0}")]
    NoIndex(u32),
    /// The index for the given track is empty.
    #[error("empty index for track {0}")]
    EmptyIndex(u32),
    /// The requested PTS is before all samples in the track.
    #[error("pts {0} is before the first sample in track {1}")]
    BeforeFirstSample(i64, u32),
}

/// Multi-track sample-accurate seeker with a per-track PTS→byte-offset index.
pub struct MultiTrackSeeker {
    /// Per-track index: track_id → sorted `Vec<SampleIndexEntry>`.
    indices: HashMap<u32, Vec<SampleIndexEntry>>,
}

impl MultiTrackSeeker {
    /// Creates an empty [`MultiTrackSeeker`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            indices: HashMap::new(),
        }
    }

    /// Builds (or replaces) the index for `track_id` from the provided sample list.
    ///
    /// # Errors
    ///
    /// This method currently always succeeds.
    pub fn build_index(
        &mut self,
        track_id: u32,
        samples: &[SampleIndexEntry],
    ) -> Result<(), MultiTrackSeekerError> {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable_by_key(|e| e.pts);
        self.indices.insert(track_id, sorted);
        Ok(())
    }

    /// Seeks to the sample-accurate position for `target_pts` within `track_id`.
    ///
    /// # Errors
    ///
    /// - [`MultiTrackSeekerError::NoIndex`] — the track has no index.
    /// - [`MultiTrackSeekerError::EmptyIndex`] — the index is empty.
    /// - [`MultiTrackSeekerError::BeforeFirstSample`] — `target_pts` is earlier
    ///   than the first indexed sample.
    pub fn seek_to_pts(
        &self,
        track_id: u32,
        target_pts: i64,
    ) -> Result<PtsSeekResult, MultiTrackSeekerError> {
        let entries = self
            .indices
            .get(&track_id)
            .ok_or(MultiTrackSeekerError::NoIndex(track_id))?;

        if entries.is_empty() {
            return Err(MultiTrackSeekerError::EmptyIndex(track_id));
        }

        let insertion = entries.partition_point(|e| e.pts <= target_pts);

        if insertion == 0 {
            return Err(MultiTrackSeekerError::BeforeFirstSample(
                target_pts, track_id,
            ));
        }

        let sample_idx = insertion - 1;
        let entry = &entries[sample_idx];

        Ok(PtsSeekResult {
            found_pts: entry.pts,
            byte_offset: entry.byte_offset,
            sample_idx,
        })
    }

    /// Returns the number of tracks that have been indexed.
    #[must_use]
    pub fn indexed_track_count(&self) -> usize {
        self.indices.len()
    }

    /// Returns the number of indexed samples for `track_id`, or `None`.
    #[must_use]
    pub fn sample_count(&self, track_id: u32) -> Option<usize> {
        self.indices.get(&track_id).map(Vec::len)
    }

    /// Clears the index for `track_id`.
    pub fn clear_index(&mut self, track_id: u32) {
        self.indices.remove(&track_id);
    }

    /// Returns a sorted slice of index entries for `track_id`, or `None`.
    #[must_use]
    pub fn entries(&self, track_id: u32) -> Option<&[SampleIndexEntry]> {
        self.indices.get(&track_id).map(Vec::as_slice)
    }
}

impl Default for MultiTrackSeeker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
