//! Segment-based parallel single-file encoding.
//!
//! Splits a single media file into GOP/keyframe-aligned segments, encodes
//! each segment concurrently using rayon, and concatenates the results in
//! order to produce a single output byte buffer.
//!
//! # Design
//!
//! - [`SegmentPlan`] describes where segment boundaries fall (by frame
//!   position).  Build one from a fixed GOP length via
//!   [`SegmentPlan::from_gop_size`].
//! - [`encode_segments_parallel`] drives encoding.  The caller supplies a
//!   codec-agnostic closure `encode_fn(index, start, end) -> Vec<u8>` that is
//!   invoked concurrently for every segment.
//! - [`concat_segments`] merges the produced [`EncodedSegment`] values into a
//!   single byte buffer, sorted by [`EncodedSegment::index`] to guarantee
//!   correct ordering regardless of rayon scheduling.

use rayon::prelude::*;

// ── FramePos ─────────────────────────────────────────────────────────────────

/// An absolute frame position within a media stream (zero-based).
///
/// Used to mark the start and end of each [`SegmentPlan`] boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FramePos(pub u64);

// ── SegmentPlan ───────────────────────────────────────────────────────────────

/// A plan that splits an input stream into segments at keyframe boundaries.
///
/// The first boundary is always `FramePos(0)`.  Subsequent boundaries are
/// placed at every keyframe (or at fixed GOP intervals when constructed via
/// [`Self::from_gop_size`]).
#[derive(Debug, Clone)]
pub struct SegmentPlan {
    /// Sorted, unique frame positions where segments start.
    ///
    /// The first element is always `FramePos(0)`.
    pub boundaries: Vec<FramePos>,
}

impl SegmentPlan {
    /// Build a plan from a fixed GOP size (a keyframe every `gop_size` frames).
    ///
    /// # Panics (in debug)
    ///
    /// Panics if `gop_size` is zero.
    #[must_use]
    pub fn from_gop_size(total_frames: u64, gop_size: u64) -> Self {
        debug_assert!(gop_size > 0, "gop_size must be > 0");
        let gop_size = gop_size.max(1);
        let boundaries = (0..total_frames)
            .step_by(gop_size as usize)
            .map(FramePos)
            .collect();
        Self { boundaries }
    }

    /// Number of segments described by this plan.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.boundaries.len()
    }

    /// Returns the half-open frame range `[start, end)` for segment `i`.
    ///
    /// The end of the last segment equals `total_frames`.
    #[must_use]
    pub fn segment_range(&self, i: usize, total_frames: u64) -> (FramePos, FramePos) {
        let start = self.boundaries[i];
        let end = self
            .boundaries
            .get(i + 1)
            .copied()
            .unwrap_or(FramePos(total_frames));
        (start, end)
    }
}

// ── EncodedSegment ────────────────────────────────────────────────────────────

/// A single encoded segment produced by [`encode_segments_parallel`].
#[derive(Debug)]
pub struct EncodedSegment {
    /// The zero-based index of this segment within the plan.
    pub index: usize,
    /// The raw encoded bytes for this segment.
    pub data: Vec<u8>,
    /// The number of frames contained in this segment.
    pub frame_count: u64,
}

// ── encode_segments_parallel ──────────────────────────────────────────────────

/// Encode `plan.segment_count()` segments in parallel, returning one
/// [`EncodedSegment`] per segment.
///
/// `encode_fn` is called concurrently (via rayon) for each segment.  It
/// receives `(segment_index, start_frame, end_frame)` and must return the
/// raw encoded bytes for that segment.  The closure must be `Send + Sync`.
///
/// The returned `Vec` is **sorted by `EncodedSegment::index`** (ascending).
/// rayon's indexed parallel iterators already preserve order, but the sort
/// is included as an explicit safety guarantee.
///
/// # Arguments
///
/// * `plan`          – segment boundary plan
/// * `total_frames`  – total frame count in the source stream
/// * `encode_fn`     – codec-agnostic encoder: `Fn(index, start, end) -> Vec<u8>`
pub fn encode_segments_parallel<F>(
    plan: &SegmentPlan,
    total_frames: u64,
    encode_fn: F,
) -> Vec<EncodedSegment>
where
    F: Fn(usize, FramePos, FramePos) -> Vec<u8> + Send + Sync,
{
    let mut segments: Vec<EncodedSegment> = (0..plan.segment_count())
        .into_par_iter()
        .map(|i| {
            let (start, end) = plan.segment_range(i, total_frames);
            let data = encode_fn(i, start, end);
            let frame_count = end.0.saturating_sub(start.0);
            EncodedSegment {
                index: i,
                data,
                frame_count,
            }
        })
        .collect();

    // Sort by index as a safety measure (rayon guarantees order for indexed
    // iterators, but an explicit sort makes the invariant unmistakable).
    segments.sort_by_key(|s| s.index);
    segments
}

// ── concat_segments ───────────────────────────────────────────────────────────

/// Concatenate [`EncodedSegment`] values into a single byte buffer.
///
/// Segments are sorted by [`EncodedSegment::index`] before concatenation so
/// callers that receive an unsorted collection still get a correct result.
#[must_use]
pub fn concat_segments(mut segments: Vec<EncodedSegment>) -> Vec<u8> {
    segments.sort_by_key(|s| s.index);
    let total: usize = segments.iter().map(|s| s.data.len()).sum();
    let mut out = Vec::with_capacity(total);
    for seg in segments {
        out.extend_from_slice(&seg.data);
    }
    out
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SegmentPlan tests ─────────────────────────────────────────────────────

    /// A plan built with GOP=10 on 30 frames must have exactly 3 boundaries
    /// at {0, 10, 20} and the three segment ranges must cover [0,10),
    /// [10,20), [20,30) with no gap or overlap.
    #[test]
    fn test_segment_plan_boundaries_cover_full_range() {
        let plan = SegmentPlan::from_gop_size(30, 10);
        assert_eq!(plan.segment_count(), 3);
        assert_eq!(plan.boundaries[0], FramePos(0));
        assert_eq!(plan.boundaries[1], FramePos(10));
        assert_eq!(plan.boundaries[2], FramePos(20));

        let (s0, e0) = plan.segment_range(0, 30);
        let (s1, e1) = plan.segment_range(1, 30);
        let (s2, e2) = plan.segment_range(2, 30);

        assert_eq!((s0, e0), (FramePos(0), FramePos(10)));
        assert_eq!((s1, e1), (FramePos(10), FramePos(20)));
        assert_eq!((s2, e2), (FramePos(20), FramePos(30)));

        // Verify full coverage with no gap or overlap
        assert_eq!(e0, s1, "gap or overlap between segment 0 and 1");
        assert_eq!(e1, s2, "gap or overlap between segment 1 and 2");
        assert_eq!(e2.0, 30, "last segment end must equal total_frames");
    }

    /// encode_segments_parallel on 5 segments where encode_fn embeds its own
    /// index as the first byte must produce segments whose first bytes are
    /// 0, 1, 2, 3, 4 in that order after concat.
    #[test]
    fn test_segment_parallel_ordering() {
        let plan = SegmentPlan::from_gop_size(50, 10);
        assert_eq!(plan.segment_count(), 5);

        let segments = encode_segments_parallel(&plan, 50, |i, _start, _end| {
            // Encode index as a single byte to verify ordering
            vec![i as u8]
        });

        let combined = concat_segments(segments);
        assert_eq!(combined, vec![0u8, 1, 2, 3, 4]);
    }

    /// Parallel and sequential concatenation must produce identical output
    /// when a deterministic passthrough encoder is used.
    ///
    /// The stub encoder serialises (index, start, end) as 3× little-endian
    /// u64 = 24 bytes per segment.
    #[test]
    fn test_segment_concat_equals_sequential() {
        let total_frames: u64 = 30;
        let gop_size: u64 = 10;
        let plan = SegmentPlan::from_gop_size(total_frames, gop_size);

        // Passthrough encoder: serialise (index, start, end) deterministically
        let encode_fn = |i: usize, start: FramePos, end: FramePos| -> Vec<u8> {
            let mut buf = Vec::with_capacity(24);
            buf.extend_from_slice(&(i as u64).to_le_bytes());
            buf.extend_from_slice(&start.0.to_le_bytes());
            buf.extend_from_slice(&end.0.to_le_bytes());
            buf
        };

        // Parallel path
        let parallel_result = {
            let segments = encode_segments_parallel(&plan, total_frames, encode_fn);
            concat_segments(segments)
        };

        // Sequential reference path
        let sequential_result: Vec<u8> = (0..plan.segment_count())
            .flat_map(|i| {
                let (start, end) = plan.segment_range(i, total_frames);
                encode_fn(i, start, end)
            })
            .collect();

        assert_eq!(
            parallel_result, sequential_result,
            "parallel and sequential results must be identical"
        );
    }

    /// A plan with a single boundary (total_frames == gop_size) must return
    /// exactly one segment that covers the full range [0, total_frames).
    #[test]
    fn test_segment_single_segment() {
        let total_frames: u64 = 10;
        let plan = SegmentPlan::from_gop_size(total_frames, total_frames);

        assert_eq!(plan.segment_count(), 1);
        assert_eq!(plan.boundaries[0], FramePos(0));

        let (start, end) = plan.segment_range(0, total_frames);
        assert_eq!(start, FramePos(0));
        assert_eq!(end, FramePos(total_frames));

        let segments = encode_segments_parallel(&plan, total_frames, |_i, _s, _e| vec![0xFFu8]);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].frame_count, total_frames);
    }

    /// 25 frames with GOP=10: boundaries at {0, 10, 20}; the last segment
    /// must cover [20, 25) — only 5 frames, not 10.
    #[test]
    fn test_segment_uneven_last_segment() {
        let total_frames: u64 = 25;
        let plan = SegmentPlan::from_gop_size(total_frames, 10);

        assert_eq!(plan.segment_count(), 3);

        let (start2, end2) = plan.segment_range(2, total_frames);
        assert_eq!(start2, FramePos(20));
        assert_eq!(end2, FramePos(25));

        let segments = encode_segments_parallel(&plan, total_frames, |_i, start, end| {
            // Return frame_count as a single little-endian u64
            let count = end.0.saturating_sub(start.0);
            count.to_le_bytes().to_vec()
        });

        assert_eq!(segments.len(), 3);

        // Last segment should have 5 frames
        let last = segments
            .iter()
            .find(|s| s.index == 2)
            .expect("segment 2 missing");
        assert_eq!(last.frame_count, 5);
    }
}
