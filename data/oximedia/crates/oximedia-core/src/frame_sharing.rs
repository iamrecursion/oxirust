//! Zero-copy frame sharing via ref-counted buffers.
//!
//! This module provides [`SharedFrame`] and [`SharedFrameBuffer`], which allow
//! multiple consumers to hold immutable references to the same frame data without
//! copying it.  Frame data is stored in an [`Arc`]-backed buffer; cloning a
//! `SharedFrame` increments the reference count rather than copying bytes.
//!
//! # Key types
//!
//! - [`SharedFrameBuffer`] — the owned backing buffer (cheaply clonable via `Arc`).
//! - [`SharedFrame`] — a view into a `SharedFrameBuffer` with metadata.
//! - [`SharedFrameSlice`] — a sub-region view into a `SharedFrameBuffer`.
//! - [`MutableFrameBuffer`] — an exclusive, mutable frame buffer that can be
//!   "frozen" into a `SharedFrameBuffer` for zero-copy distribution.
//!
//! # Example
//!
//! ```
//! use oximedia_core::frame_sharing::{MutableFrameBuffer, SharedFrame, FrameMetadata};
//! use oximedia_core::types::{PixelFormat, Timestamp, Rational};
//!
//! // Encode a frame into a mutable buffer, then share it.
//! let mut mbuf = MutableFrameBuffer::new(1920 * 1080 * 3);
//! mbuf.data_mut().fill(0x42);
//! let shared = mbuf.freeze();
//!
//! let meta = FrameMetadata {
//!     width: 1920,
//!     height: 1080,
//!     format: PixelFormat::Rgb24,
//!     timestamp: Timestamp::new(0, Rational::new(1, 30)),
//! };
//! let frame = SharedFrame::new(shared, meta);
//! let clone = frame.clone(); // zero-copy — shares the same Arc<[u8]>
//! assert_eq!(frame.ref_count(), 2);
//! assert_eq!(clone.data()[0], 0x42);
//! ```

#![allow(dead_code)]

use std::sync::Arc;

use crate::types::{PixelFormat, Timestamp};

// ─────────────────────────────────────────────────────────────────────────────
// FrameMetadata
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata describing the content and timing of a video frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameMetadata {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format of the frame data.
    pub format: PixelFormat,
    /// Presentation timestamp of this frame.
    pub timestamp: Timestamp,
}

// ─────────────────────────────────────────────────────────────────────────────
// SharedFrameBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// An immutable, ref-counted backing buffer for frame data.
///
/// Multiple [`SharedFrame`] and [`SharedFrameSlice`] objects may reference the
/// same `SharedFrameBuffer` without copying bytes.  The buffer is freed when
/// the last reference is dropped.
#[derive(Debug, Clone)]
pub struct SharedFrameBuffer {
    inner: Arc<[u8]>,
}

impl SharedFrameBuffer {
    /// Creates a `SharedFrameBuffer` from an existing byte slice, copying the data once.
    #[must_use]
    pub fn from_slice(data: &[u8]) -> Self {
        Self {
            inner: Arc::from(data),
        }
    }

    /// Creates a `SharedFrameBuffer` from a `Vec<u8>` without copying.
    #[must_use]
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self {
            inner: Arc::from(data.into_boxed_slice()),
        }
    }

    /// Returns a read-only view of the buffer contents.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.inner
    }

    /// Returns the total length of the buffer in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of strong references currently pointing to this buffer.
    ///
    /// This includes the current `SharedFrameBuffer` itself.
    #[must_use]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Creates a [`SharedFrameSlice`] covering a sub-range of this buffer.
    ///
    /// # Panics
    ///
    /// Panics if `start > end` or `end > self.len()`.
    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> SharedFrameSlice {
        assert!(start <= end, "slice start must be <= end");
        assert!(end <= self.len(), "slice end out of bounds");
        SharedFrameSlice {
            buffer: self.clone(),
            start,
            end,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SharedFrameSlice
// ─────────────────────────────────────────────────────────────────────────────

/// A zero-copy sub-region view into a [`SharedFrameBuffer`].
///
/// Does not copy data; retains an `Arc` reference to the backing buffer.
#[derive(Debug, Clone)]
pub struct SharedFrameSlice {
    buffer: SharedFrameBuffer,
    start: usize,
    end: usize,
}

impl SharedFrameSlice {
    /// Returns the slice data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.buffer.data()[self.start..self.end]
    }

    /// Returns the length of this slice in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Returns `true` if the slice is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Returns the byte offset within the backing buffer where this slice starts.
    #[must_use]
    pub fn offset(&self) -> usize {
        self.start
    }

    /// Returns the number of strong references to the backing buffer.
    #[must_use]
    pub fn ref_count(&self) -> usize {
        self.buffer.ref_count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MutableFrameBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// An exclusive, mutable frame buffer.
///
/// Used to construct frame data before publishing it for zero-copy sharing.
/// Call [`freeze`](MutableFrameBuffer::freeze) to convert into a
/// `SharedFrameBuffer` with zero copies.
#[derive(Debug)]
pub struct MutableFrameBuffer {
    data: Vec<u8>,
}

impl MutableFrameBuffer {
    /// Creates a new zeroed mutable buffer of `size` bytes.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
        }
    }

    /// Creates a mutable buffer from an existing `Vec<u8>`.
    #[must_use]
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Returns a read-only view of the buffer.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns a mutable view of the buffer.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Returns the length of the buffer in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Converts this mutable buffer into an immutable [`SharedFrameBuffer`]
    /// without copying the underlying allocation.
    #[must_use]
    pub fn freeze(self) -> SharedFrameBuffer {
        SharedFrameBuffer::from_vec(self.data)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SharedFrame
// ─────────────────────────────────────────────────────────────────────────────

/// A zero-copy, ref-counted video frame.
///
/// Cloning a `SharedFrame` shares the underlying buffer via `Arc`; no pixel
/// data is copied.
#[derive(Debug, Clone)]
pub struct SharedFrame {
    buffer: SharedFrameBuffer,
    /// Frame metadata (dimensions, format, timing).
    pub meta: FrameMetadata,
}

impl SharedFrame {
    /// Creates a new `SharedFrame` wrapping `buffer` with the given metadata.
    #[must_use]
    pub fn new(buffer: SharedFrameBuffer, meta: FrameMetadata) -> Self {
        Self { buffer, meta }
    }

    /// Returns a read-only reference to the pixel data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.buffer.data()
    }

    /// Returns the total size of the frame data in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the frame buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns the number of strong references currently held to the backing buffer.
    ///
    /// Useful for tracking how many consumers are sharing this frame.
    #[must_use]
    pub fn ref_count(&self) -> usize {
        self.buffer.ref_count()
    }

    /// Returns a reference to the underlying [`SharedFrameBuffer`].
    #[must_use]
    pub fn buffer(&self) -> &SharedFrameBuffer {
        &self.buffer
    }

    /// Returns a sub-region slice of this frame's backing buffer.
    ///
    /// # Panics
    ///
    /// Panics if `start > end` or `end > self.len()`.
    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> SharedFrameSlice {
        self.buffer.slice(start, end)
    }

    /// Attempts to obtain exclusive ownership of the backing buffer, returning
    /// a [`MutableFrameBuffer`] if this is the only strong reference.
    ///
    /// Returns `Err(self)` if more than one reference exists.
    pub fn try_into_mutable(self) -> Result<MutableFrameBuffer, Self> {
        // Arc<[u8]> is an unsized type; try_unwrap requires Sized.
        // Instead, check the ref count: if we are the sole owner, copy the data
        // into a new Vec<u8> for the MutableFrameBuffer.
        if Arc::strong_count(&self.buffer.inner) == 1 {
            // We are the only reference — safe to treat as exclusive.
            let data = self.buffer.inner.to_vec();
            Ok(MutableFrameBuffer::from_vec(data))
        } else {
            Err(self)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PixelFormat, Rational, Timestamp};

    fn make_meta() -> FrameMetadata {
        FrameMetadata {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb24,
            timestamp: Timestamp::new(0, Rational::new(1, 30)),
        }
    }

    // 1. SharedFrameBuffer from_slice
    #[test]
    fn test_shared_buffer_from_slice() {
        let buf = SharedFrameBuffer::from_slice(&[1, 2, 3, 4]);
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.data(), &[1, 2, 3, 4]);
    }

    // 2. SharedFrameBuffer from_vec – no copy
    #[test]
    fn test_shared_buffer_from_vec() {
        let v = vec![10u8, 20, 30];
        let buf = SharedFrameBuffer::from_vec(v);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.data()[1], 20);
    }

    // 3. Cloning SharedFrameBuffer increases ref_count
    #[test]
    fn test_shared_buffer_ref_count() {
        let buf = SharedFrameBuffer::from_slice(&[0u8; 16]);
        assert_eq!(buf.ref_count(), 1);
        let clone = buf.clone();
        assert_eq!(buf.ref_count(), 2);
        assert_eq!(clone.ref_count(), 2);
        drop(clone);
        assert_eq!(buf.ref_count(), 1);
    }

    // 4. SharedFrameSlice covers correct sub-range
    #[test]
    fn test_shared_buffer_slice() {
        let buf = SharedFrameBuffer::from_slice(&[0, 1, 2, 3, 4, 5]);
        let slice = buf.slice(2, 5);
        assert_eq!(slice.data(), &[2, 3, 4]);
        assert_eq!(slice.len(), 3);
        assert_eq!(slice.offset(), 2);
    }

    // 5. SharedFrameSlice shares the backing buffer ref-count
    #[test]
    fn test_slice_shares_ref_count() {
        let buf = SharedFrameBuffer::from_slice(&[0u8; 8]);
        let slice = buf.slice(0, 4);
        assert_eq!(buf.ref_count(), 2);
        drop(slice);
        assert_eq!(buf.ref_count(), 1);
    }

    // 6. Empty slice
    #[test]
    fn test_empty_slice() {
        let buf = SharedFrameBuffer::from_slice(&[1, 2, 3]);
        let slice = buf.slice(1, 1);
        assert!(slice.is_empty());
        assert_eq!(slice.len(), 0);
    }

    // 7. MutableFrameBuffer zeroed on new
    #[test]
    fn test_mutable_buffer_new_zeroed() {
        let mbuf = MutableFrameBuffer::new(8);
        assert_eq!(mbuf.len(), 8);
        assert!(mbuf.data().iter().all(|&b| b == 0));
    }

    // 8. MutableFrameBuffer write then freeze
    #[test]
    fn test_mutable_buffer_freeze() {
        let mut mbuf = MutableFrameBuffer::new(4);
        mbuf.data_mut().copy_from_slice(&[10, 20, 30, 40]);
        let shared = mbuf.freeze();
        assert_eq!(shared.data(), &[10, 20, 30, 40]);
        assert_eq!(shared.ref_count(), 1);
    }

    // 9. SharedFrame construction
    #[test]
    fn test_shared_frame_new() {
        let buf = SharedFrameBuffer::from_slice(&[1u8; 48]); // 4×4 RGB24
        let frame = SharedFrame::new(buf, make_meta());
        assert_eq!(frame.len(), 48);
        assert_eq!(frame.meta.width, 4);
    }

    // 10. Cloning SharedFrame does not copy data
    #[test]
    fn test_shared_frame_clone_no_copy() {
        let mut mbuf = MutableFrameBuffer::new(12);
        mbuf.data_mut().fill(0xAB);
        let shared_buf = mbuf.freeze();
        let frame = SharedFrame::new(shared_buf, make_meta());

        assert_eq!(frame.ref_count(), 1);
        let clone = frame.clone();
        assert_eq!(frame.ref_count(), 2);
        assert_eq!(clone.ref_count(), 2);
        assert_eq!(clone.data()[0], 0xAB);
    }

    // 11. SharedFrame::slice
    #[test]
    fn test_shared_frame_slice() {
        let data: Vec<u8> = (0..12).collect();
        let buf = SharedFrameBuffer::from_vec(data);
        let frame = SharedFrame::new(buf, make_meta());
        let slice = frame.slice(4, 8);
        assert_eq!(slice.data(), &[4, 5, 6, 7]);
    }

    // 12. try_into_mutable – succeeds when single reference
    #[test]
    fn test_try_into_mutable_success() {
        let buf = SharedFrameBuffer::from_slice(&[1, 2, 3]);
        let frame = SharedFrame::new(buf, make_meta());
        let mutable = frame.try_into_mutable().expect("should succeed");
        assert_eq!(mutable.data(), &[1, 2, 3]);
    }

    // 13. try_into_mutable – fails when multiple references
    #[test]
    fn test_try_into_mutable_fails_with_clone() {
        let buf = SharedFrameBuffer::from_slice(&[7, 8, 9]);
        let frame = SharedFrame::new(buf, make_meta());
        let _clone = frame.clone();
        let result = frame.try_into_mutable();
        assert!(result.is_err());
    }

    // 14. SharedFrameBuffer is_empty
    #[test]
    fn test_shared_buffer_is_empty() {
        let empty = SharedFrameBuffer::from_slice(&[]);
        assert!(empty.is_empty());
        let nonempty = SharedFrameBuffer::from_slice(&[1]);
        assert!(!nonempty.is_empty());
    }

    // 15. MutableFrameBuffer from_vec round-trip
    #[test]
    fn test_mutable_buffer_from_vec() {
        let mbuf = MutableFrameBuffer::from_vec(vec![5, 10, 15]);
        assert_eq!(mbuf.len(), 3);
        assert_eq!(mbuf.data()[2], 15);
    }
}
