#![allow(dead_code)]
//! PTS/DTS timestamp utilities and repair helpers.
//!
//! Provides `PtsDts` for carrying presentation/decode timestamps,
//! `PtsQueue` for reordering packets, and `PtsDtsRepair` for fixing
//! common timestamp pathologies (negative DTS, wrong ordering).

/// Presentation timestamp (PTS) and decode timestamp (DTS) pair.
///
/// Both values are in the container's native time base (e.g. 90 kHz ticks).
/// Either value may be absent (`None`) when not signalled by the container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtsDts {
    /// Presentation timestamp.
    pub pts: Option<i64>,
    /// Decode timestamp.
    pub dts: Option<i64>,
}

impl PtsDts {
    /// Creates a `PtsDts` with both values present.
    #[must_use]
    pub const fn new(pts: i64, dts: i64) -> Self {
        Self {
            pts: Some(pts),
            dts: Some(dts),
        }
    }

    /// Creates a `PtsDts` with only a PTS (no DTS field in the stream).
    #[must_use]
    pub const fn pts_only(pts: i64) -> Self {
        Self {
            pts: Some(pts),
            dts: None,
        }
    }

    /// Creates an empty `PtsDts` (both absent).
    #[must_use]
    pub const fn none() -> Self {
        Self {
            pts: None,
            dts: None,
        }
    }

    /// Returns `true` when a PTS value is present.
    #[must_use]
    pub fn has_pts(&self) -> bool {
        self.pts.is_some()
    }

    /// Returns `true` when PTS and DTS are both present and equal.
    #[must_use]
    pub fn is_pts_dts_equal(&self) -> bool {
        match (self.pts, self.dts) {
            (Some(p), Some(d)) => p == d,
            _ => false,
        }
    }

    /// Returns the effective decode time: DTS when present, else PTS.
    #[must_use]
    pub fn effective_dts(&self) -> Option<i64> {
        self.dts.or(self.pts)
    }

    /// Returns `true` if the DTS is negative (a common pathology in some muxers).
    #[must_use]
    pub fn has_negative_dts(&self) -> bool {
        self.dts.is_some_and(|d| d < 0)
    }
}

/// Apply independent PTS and DTS offset corrections to a slice of [`PtsDts`]
/// values in a single cache-friendly pass.
///
/// This is the primary batch entry point for remuxing and splicing operations
/// where thousands of packet timestamps must be shifted by a constant delta
/// (e.g. when concatenating two streams, rebasing a clip to zero, or correcting
/// an initial PTS/DTS gap).
///
/// # Behaviour
///
/// - If `pts_offset` is non-zero:
///   - `Some(p)` → `Some(p + pts_offset)`
///   - `None`    → `Some(pts_offset)` (materialises a PTS when absent)
/// - If `dts_offset` is non-zero:
///   - `Some(d)` → `Some(d + dts_offset)`
///   - `None`    → `Some(dts_offset)` (materialises a DTS when absent)
/// - Zero offsets leave the corresponding field unchanged (including `None`).
///
/// # Example
///
/// ```
/// use oximedia_container::pts_dts::{PtsDts, rewrite_timestamps_batch};
///
/// let mut batch = vec![
///     PtsDts::new(0, 0),
///     PtsDts::new(3600, 3600),
///     PtsDts::pts_only(7200),
/// ];
///
/// rewrite_timestamps_batch(&mut batch, 1000, 500);
///
/// assert_eq!(batch[0].pts, Some(1000));
/// assert_eq!(batch[0].dts, Some(500));
/// assert_eq!(batch[1].pts, Some(4600));
/// assert_eq!(batch[2].pts, Some(8200));
/// // DTS was None for batch[2], so it is materialised:
/// assert_eq!(batch[2].dts, Some(500));
/// ```
pub fn rewrite_timestamps_batch(headers: &mut [PtsDts], pts_offset: i64, dts_offset: i64) {
    // Fast-path: no-op when both offsets are zero to avoid touching cache lines.
    if pts_offset == 0 && dts_offset == 0 {
        return;
    }
    for h in headers.iter_mut() {
        if pts_offset != 0 {
            h.pts = Some(h.pts.map_or(pts_offset, |p| p + pts_offset));
        }
        if dts_offset != 0 {
            h.dts = Some(h.dts.map_or(dts_offset, |d| d + dts_offset));
        }
    }
}

/// Rebase a slice of [`PtsDts`] values so that the earliest effective DTS
/// (or PTS when DTS is absent) becomes zero, shifting all other timestamps by
/// the same offset.
///
/// Returns the computed offset that was applied. Returns 0 when the slice is
/// already anchored at zero, or when it is empty.
///
/// This is the idiomatic "rebase to zero" helper for use in splicing pipelines.
pub fn rebase_timestamps_to_zero(headers: &mut [PtsDts]) -> i64 {
    let min_ts = headers
        .iter()
        .filter_map(|h| h.dts.or(h.pts))
        .min()
        .unwrap_or(0);
    if min_ts == 0 {
        return 0;
    }
    // Negative min → shift forward (positive offset).
    // Positive min → shift backward (negative offset).
    let offset = -min_ts;
    rewrite_timestamps_batch(headers, offset, offset);
    offset
}

/// A packet entry stored in the reorder queue.
#[derive(Debug, Clone)]
pub struct PtsEntry {
    /// Sequence number of the packet for stable sorting.
    pub seq: u64,
    /// Timestamp pair.
    pub ts: PtsDts,
    /// Arbitrary payload bytes (e.g. compressed frame data).
    pub data: Vec<u8>,
}

/// A small reorder queue that sorts packets by their PTS before delivery.
#[derive(Debug, Default)]
pub struct PtsQueue {
    entries: Vec<PtsEntry>,
    seq_counter: u64,
}

impl PtsQueue {
    /// Creates an empty `PtsQueue`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a packet into the queue.
    pub fn push(&mut self, ts: PtsDts, data: Vec<u8>) {
        let seq = self.seq_counter;
        self.seq_counter += 1;
        self.entries.push(PtsEntry { seq, ts, data });
    }

    /// Returns the number of entries in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns a reference to the entry with the earliest PTS (or first
    /// inserted when PTS is absent).
    #[must_use]
    pub fn earliest(&self) -> Option<&PtsEntry> {
        self.entries.iter().min_by(|a, b| {
            let pa = a.ts.pts.unwrap_or(i64::MAX);
            let pb = b.ts.pts.unwrap_or(i64::MAX);
            pa.cmp(&pb).then(a.seq.cmp(&b.seq))
        })
    }

    /// Removes and returns the entry with the earliest PTS.
    pub fn pop_earliest(&mut self) -> Option<PtsEntry> {
        if self.entries.is_empty() {
            return None;
        }
        let idx = self
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let pa = a.ts.pts.unwrap_or(i64::MAX);
                let pb = b.ts.pts.unwrap_or(i64::MAX);
                pa.cmp(&pb).then(a.seq.cmp(&b.seq))
            })
            .map(|(i, _)| i)?;
        Some(self.entries.remove(idx))
    }

    /// Sorts all entries in the queue by PTS and returns them in order,
    /// draining the queue.
    pub fn reorder(&mut self) -> Vec<PtsEntry> {
        let mut out = std::mem::take(&mut self.entries);
        out.sort_by(|a, b| {
            let pa = a.ts.pts.unwrap_or(i64::MAX);
            let pb = b.ts.pts.unwrap_or(i64::MAX);
            pa.cmp(&pb).then(a.seq.cmp(&b.seq))
        });
        out
    }
}

/// Repairs common PTS/DTS pathologies in a stream of timestamps.
#[derive(Debug, Default)]
pub struct PtsDtsRepair {
    repair_count: u64,
    dts_offset: i64,
}

impl PtsDtsRepair {
    /// Creates a new `PtsDtsRepair` instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of timestamps that have been repaired.
    #[must_use]
    pub fn repair_count(&self) -> u64 {
        self.repair_count
    }

    /// Fixes a negative DTS by shifting it to 0 and recording the offset
    /// applied for future packets.
    ///
    /// If the DTS is already non-negative the input is returned unchanged.
    pub fn fix_negative_dts(&mut self, ts: PtsDts) -> PtsDts {
        if let Some(dts) = ts.dts {
            if dts < 0 {
                let shift = -dts;
                self.dts_offset += shift;
                self.repair_count += 1;
                return PtsDts {
                    pts: ts.pts.map(|p| p + shift),
                    dts: Some(0),
                };
            }
        }
        ts
    }

    /// Applies the accumulated DTS offset to a new timestamp pair.
    /// Use this after `fix_negative_dts` to keep subsequent packets aligned.
    #[must_use]
    pub fn apply_offset(&self, ts: PtsDts) -> PtsDts {
        if self.dts_offset == 0 {
            return ts;
        }
        PtsDts {
            pts: ts.pts.map(|p| p + self.dts_offset),
            dts: ts.dts.map(|d| d + self.dts_offset),
        }
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // 1. has_pts – Some
    #[test]
    fn test_has_pts_some() {
        let ts = PtsDts::new(100, 90);
        assert!(ts.has_pts());
    }

    // 2. has_pts – None
    #[test]
    fn test_has_pts_none() {
        let ts = PtsDts::none();
        assert!(!ts.has_pts());
    }

    // 3. is_pts_dts_equal – equal
    #[test]
    fn test_is_pts_dts_equal_true() {
        let ts = PtsDts::new(200, 200);
        assert!(ts.is_pts_dts_equal());
    }

    // 4. is_pts_dts_equal – not equal
    #[test]
    fn test_is_pts_dts_equal_false() {
        let ts = PtsDts::new(200, 180);
        assert!(!ts.is_pts_dts_equal());
    }

    // 5. is_pts_dts_equal – DTS absent
    #[test]
    fn test_is_pts_dts_equal_no_dts() {
        let ts = PtsDts::pts_only(200);
        assert!(!ts.is_pts_dts_equal());
    }

    // 6. effective_dts prefers DTS
    #[test]
    fn test_effective_dts_prefers_dts() {
        let ts = PtsDts::new(200, 180);
        assert_eq!(ts.effective_dts(), Some(180));
    }

    // 7. effective_dts falls back to PTS
    #[test]
    fn test_effective_dts_falls_back_to_pts() {
        let ts = PtsDts::pts_only(200);
        assert_eq!(ts.effective_dts(), Some(200));
    }

    // 8. has_negative_dts
    #[test]
    fn test_has_negative_dts() {
        let ts = PtsDts::new(0, -90);
        assert!(ts.has_negative_dts());
    }

    // 9. PtsQueue empty
    #[test]
    fn test_queue_empty() {
        let q = PtsQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    // 10. push / len
    #[test]
    fn test_queue_push_len() {
        let mut q = PtsQueue::new();
        q.push(PtsDts::new(100, 100), vec![1, 2, 3]);
        assert_eq!(q.len(), 1);
    }

    // 11. earliest returns smallest PTS
    #[test]
    fn test_queue_earliest() {
        let mut q = PtsQueue::new();
        q.push(PtsDts::new(300, 300), vec![]);
        q.push(PtsDts::new(100, 100), vec![]);
        q.push(PtsDts::new(200, 200), vec![]);
        assert_eq!(
            q.earliest().expect("operation should succeed").ts.pts,
            Some(100)
        );
    }

    // 12. reorder drains and sorts
    #[test]
    fn test_queue_reorder() {
        let mut q = PtsQueue::new();
        q.push(PtsDts::new(300, 300), vec![]);
        q.push(PtsDts::new(100, 100), vec![]);
        q.push(PtsDts::new(200, 200), vec![]);
        let sorted = q.reorder();
        let pts_values: Vec<i64> = sorted
            .iter()
            .map(|e| e.ts.pts.expect("operation should succeed"))
            .collect();
        assert_eq!(pts_values, vec![100, 200, 300]);
        assert!(q.is_empty());
    }

    // 13. fix_negative_dts shifts to zero
    #[test]
    fn test_fix_negative_dts() {
        let mut repair = PtsDtsRepair::new();
        let ts = PtsDts::new(0, -180);
        let fixed = repair.fix_negative_dts(ts);
        assert_eq!(fixed.dts, Some(0));
        assert_eq!(repair.repair_count(), 1);
    }

    // 14. fix_negative_dts leaves positive DTS untouched
    #[test]
    fn test_fix_negative_dts_no_op() {
        let mut repair = PtsDtsRepair::new();
        let ts = PtsDts::new(100, 90);
        let fixed = repair.fix_negative_dts(ts);
        assert_eq!(fixed.dts, Some(90));
        assert_eq!(repair.repair_count(), 0);
    }

    // ─── Batch function tests ─────────────────────────────────────────────────

    // 15. rewrite_timestamps_batch applies pts and dts offsets to all entries
    #[test]
    fn test_rewrite_timestamps_batch_both_offsets() {
        let mut batch = vec![
            PtsDts::new(0, 0),
            PtsDts::new(3600, 3600),
            PtsDts::new(7200, 7200),
        ];
        rewrite_timestamps_batch(&mut batch, 1000, 500);
        assert_eq!(batch[0].pts, Some(1000));
        assert_eq!(batch[0].dts, Some(500));
        assert_eq!(batch[1].pts, Some(4600));
        assert_eq!(batch[1].dts, Some(4100));
        assert_eq!(batch[2].pts, Some(8200));
        assert_eq!(batch[2].dts, Some(7700));
    }

    // 16. rewrite_timestamps_batch materialises PTS when None
    #[test]
    fn test_rewrite_timestamps_batch_materialises_pts() {
        let mut batch = vec![PtsDts::none()];
        rewrite_timestamps_batch(&mut batch, 500, 0);
        assert_eq!(batch[0].pts, Some(500));
        assert_eq!(batch[0].dts, None); // dts_offset == 0, so None stays None
    }

    // 17. rewrite_timestamps_batch materialises DTS when None
    #[test]
    fn test_rewrite_timestamps_batch_materialises_dts() {
        let mut batch = vec![PtsDts::pts_only(1000)];
        rewrite_timestamps_batch(&mut batch, 0, 200);
        assert_eq!(batch[0].pts, Some(1000)); // pts_offset == 0 → unchanged
        assert_eq!(batch[0].dts, Some(200));
    }

    // 18. rewrite_timestamps_batch no-op when both offsets are zero
    #[test]
    fn test_rewrite_timestamps_batch_zero_offsets_noop() {
        let mut batch = vec![PtsDts::new(100, 90), PtsDts::none()];
        rewrite_timestamps_batch(&mut batch, 0, 0);
        assert_eq!(batch[0].pts, Some(100));
        assert_eq!(batch[0].dts, Some(90));
        assert_eq!(batch[1].pts, None);
        assert_eq!(batch[1].dts, None);
    }

    // 19. rewrite_timestamps_batch with negative offset
    #[test]
    fn test_rewrite_timestamps_batch_negative_offset() {
        let mut batch = vec![PtsDts::new(5000, 5000), PtsDts::new(10000, 10000)];
        rewrite_timestamps_batch(&mut batch, -1000, -1000);
        assert_eq!(batch[0].pts, Some(4000));
        assert_eq!(batch[0].dts, Some(4000));
        assert_eq!(batch[1].pts, Some(9000));
    }

    // 20. rewrite_timestamps_batch on empty slice is a no-op
    #[test]
    fn test_rewrite_timestamps_batch_empty() {
        let mut batch: Vec<PtsDts> = Vec::new();
        rewrite_timestamps_batch(&mut batch, 999, 999);
        assert!(batch.is_empty());
    }

    // 21. rebase_timestamps_to_zero shifts positive earliest DTS to zero
    #[test]
    fn test_rebase_timestamps_to_zero_positive_min() {
        let mut batch = vec![PtsDts::new(5000, 4500), PtsDts::new(10000, 9500)];
        let offset = rebase_timestamps_to_zero(&mut batch);
        assert_eq!(offset, -4500);
        assert_eq!(batch[0].dts, Some(0));
        assert_eq!(batch[0].pts, Some(500));
        assert_eq!(batch[1].dts, Some(5000));
    }

    // 22. rebase_timestamps_to_zero shifts negative earliest DTS to zero
    #[test]
    fn test_rebase_timestamps_to_zero_negative_min() {
        let mut batch = vec![PtsDts::new(0, -1000), PtsDts::new(3600, 2600)];
        let offset = rebase_timestamps_to_zero(&mut batch);
        assert_eq!(offset, 1000);
        assert_eq!(batch[0].dts, Some(0));
        assert_eq!(batch[0].pts, Some(1000));
        assert_eq!(batch[1].dts, Some(3600));
    }

    // 23. rebase_timestamps_to_zero on already-zero batch returns 0
    #[test]
    fn test_rebase_timestamps_to_zero_already_zero() {
        let mut batch = vec![PtsDts::new(0, 0), PtsDts::new(3600, 3600)];
        let offset = rebase_timestamps_to_zero(&mut batch);
        assert_eq!(offset, 0);
        assert_eq!(batch[0].dts, Some(0));
    }

    // 24. rebase_timestamps_to_zero falls back to PTS when DTS absent
    #[test]
    fn test_rebase_timestamps_to_zero_uses_pts_when_dts_absent() {
        let mut batch = vec![PtsDts::pts_only(2000), PtsDts::pts_only(5000)];
        let offset = rebase_timestamps_to_zero(&mut batch);
        assert_eq!(offset, -2000);
        assert_eq!(batch[0].pts, Some(0));
        assert_eq!(batch[0].dts, Some(-2000)); // materialised dts = None + offset
    }

    // 25. rebase_timestamps_to_zero on empty slice returns 0
    #[test]
    fn test_rebase_timestamps_to_zero_empty() {
        let mut batch: Vec<PtsDts> = Vec::new();
        let offset = rebase_timestamps_to_zero(&mut batch);
        assert_eq!(offset, 0);
    }
}
