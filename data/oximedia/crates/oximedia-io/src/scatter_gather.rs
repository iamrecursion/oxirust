//! Scatter-gather (vectored) I/O primitives.
//!
//! Provides building blocks for assembling discontiguous buffers, coalescing
//! sequential I/O vectors, and filling pre-allocated read buffers.

#![allow(dead_code)]

// ──────────────────────────────────────────────────────────────────────────────
// IoVec
// ──────────────────────────────────────────────────────────────────────────────

/// A single scatter/gather I/O buffer with an associated file offset.
#[derive(Debug, Clone)]
pub struct IoVec {
    /// Data bytes for this vector element.
    pub data: Vec<u8>,
    /// File offset at which this data begins.
    pub offset: u64,
}

impl IoVec {
    /// Create a new `IoVec`.
    #[must_use]
    pub fn new(data: Vec<u8>, offset: u64) -> Self {
        Self { data, offset }
    }

    /// Number of bytes in this vector element.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if this vector element contains no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ScatterGatherList
// ──────────────────────────────────────────────────────────────────────────────

/// An ordered list of [`IoVec`] elements used to describe scatter/gather I/O.
#[derive(Debug, Default)]
pub struct ScatterGatherList {
    /// Component I/O vectors.
    pub iovec: Vec<IoVec>,
    /// Cached total byte count across all vectors.
    pub total_bytes: u64,
}

impl ScatterGatherList {
    /// Create an empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an I/O vector.
    pub fn add(&mut self, data: Vec<u8>, offset: u64) {
        self.total_bytes += data.len() as u64;
        self.iovec.push(IoVec::new(data, offset));
    }

    /// Total number of bytes across all vectors.
    #[must_use]
    pub fn total_len(&self) -> u64 {
        self.total_bytes
    }

    /// Merge all vectors into a single contiguous buffer.
    ///
    /// The vectors are appended in the order they were added; their offsets
    /// are not used for sorting.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn consolidate(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.total_bytes as usize);
        for v in &self.iovec {
            out.extend_from_slice(&v.data);
        }
        out
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ReadVec
// ──────────────────────────────────────────────────────────────────────────────

/// A collection of pre-allocated receive buffers for scatter reads.
#[derive(Debug, Default)]
pub struct ReadVec {
    /// Individual receive buffers.
    pub buffers: Vec<Vec<u8>>,
}

impl ReadVec {
    /// Create a `ReadVec` with the given pre-allocated buffers.
    #[must_use]
    pub fn new(buffers: Vec<Vec<u8>>) -> Self {
        Self { buffers }
    }

    /// Sum of `len()` across all buffers (bytes allocated, not used).
    pub fn total_capacity(&self) -> usize {
        self.buffers.iter().map(Vec::len).sum()
    }

    /// Copy bytes from `src` into the buffers in sequence.
    ///
    /// Returns the number of bytes actually copied.
    pub fn fill_from(&mut self, src: &[u8]) -> usize {
        let mut pos = 0usize;
        for buf in &mut self.buffers {
            if pos >= src.len() {
                break;
            }
            let space = buf.len();
            let available = src.len() - pos;
            let to_copy = space.min(available);
            buf[..to_copy].copy_from_slice(&src[pos..pos + to_copy]);
            pos += to_copy;
        }
        pos
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// WriteVec
// ──────────────────────────────────────────────────────────────────────────────

/// A collection of [`IoVec`] elements representing a scatter write.
#[derive(Debug, Default)]
pub struct WriteVec {
    /// The I/O vectors to be written.
    pub buffers: Vec<IoVec>,
}

impl WriteVec {
    /// Create an empty `WriteVec`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an I/O vector.
    pub fn push(&mut self, iov: IoVec) {
        self.buffers.push(iov);
    }

    /// Merge adjacent (sequentially contiguous) I/O vectors into single entries.
    ///
    /// Two vectors are considered adjacent when `a.offset + a.len() == b.offset`.
    #[must_use]
    pub fn coalesce_sequential(&self) -> Vec<IoVec> {
        let mut result: Vec<IoVec> = Vec::new();

        for iov in &self.buffers {
            if let Some(last) = result.last_mut() {
                let last_end = last.offset + last.data.len() as u64;
                if last_end == iov.offset {
                    // Adjacent — extend.
                    last.data.extend_from_slice(&iov.data);
                    continue;
                }
            }
            result.push(iov.clone());
        }

        result
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // IoVec ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_iovec_len() {
        let v = IoVec::new(vec![1, 2, 3], 0);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_iovec_is_empty_true() {
        let v = IoVec::new(vec![], 0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_iovec_is_empty_false() {
        let v = IoVec::new(vec![0], 0);
        assert!(!v.is_empty());
    }

    // ScatterGatherList ───────────────────────────────────────────────────────

    #[test]
    fn test_sgl_add_updates_total() {
        let mut sgl = ScatterGatherList::new();
        sgl.add(vec![1, 2, 3], 0);
        sgl.add(vec![4, 5], 3);
        assert_eq!(sgl.total_len(), 5);
    }

    #[test]
    fn test_sgl_consolidate_order() {
        let mut sgl = ScatterGatherList::new();
        sgl.add(vec![1, 2], 0);
        sgl.add(vec![3, 4], 2);
        assert_eq!(sgl.consolidate(), vec![1u8, 2, 3, 4]);
    }

    #[test]
    fn test_sgl_consolidate_empty() {
        let sgl = ScatterGatherList::new();
        assert!(sgl.consolidate().is_empty());
    }

    #[test]
    fn test_sgl_total_len_empty() {
        let sgl = ScatterGatherList::new();
        assert_eq!(sgl.total_len(), 0);
    }

    // ReadVec ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_readvec_total_capacity() {
        let rv = ReadVec::new(vec![vec![0u8; 10], vec![0u8; 20]]);
        assert_eq!(rv.total_capacity(), 30);
    }

    #[test]
    fn test_readvec_fill_from_fits() {
        let mut rv = ReadVec::new(vec![vec![0u8; 4], vec![0u8; 4]]);
        let copied = rv.fill_from(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(copied, 8);
        assert_eq!(rv.buffers[0], vec![1, 2, 3, 4]);
        assert_eq!(rv.buffers[1], vec![5, 6, 7, 8]);
    }

    #[test]
    fn test_readvec_fill_from_partial() {
        let mut rv = ReadVec::new(vec![vec![0u8; 8]]);
        let copied = rv.fill_from(&[1, 2, 3]);
        assert_eq!(copied, 3);
        assert_eq!(&rv.buffers[0][..3], &[1u8, 2, 3]);
    }

    #[test]
    fn test_readvec_fill_from_empty_src() {
        let mut rv = ReadVec::new(vec![vec![0u8; 4]]);
        let copied = rv.fill_from(&[]);
        assert_eq!(copied, 0);
    }

    // WriteVec ────────────────────────────────────────────────────────────────

    #[test]
    fn test_writevec_coalesce_adjacent() {
        let mut wv = WriteVec::new();
        wv.push(IoVec::new(vec![1, 2], 0));
        wv.push(IoVec::new(vec![3, 4], 2));
        let merged = wv.coalesce_sequential();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].data, vec![1u8, 2, 3, 4]);
        assert_eq!(merged[0].offset, 0);
    }

    #[test]
    fn test_writevec_coalesce_non_adjacent() {
        let mut wv = WriteVec::new();
        wv.push(IoVec::new(vec![1, 2], 0));
        wv.push(IoVec::new(vec![3, 4], 10)); // gap at offsets 2-9
        let merged = wv.coalesce_sequential();
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_writevec_coalesce_empty() {
        let wv = WriteVec::new();
        assert!(wv.coalesce_sequential().is_empty());
    }

    #[test]
    fn test_writevec_coalesce_three_adjacent() {
        let mut wv = WriteVec::new();
        wv.push(IoVec::new(vec![1], 0));
        wv.push(IoVec::new(vec![2], 1));
        wv.push(IoVec::new(vec![3], 2));
        let merged = wv.coalesce_sequential();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].data, vec![1u8, 2, 3]);
    }
}
