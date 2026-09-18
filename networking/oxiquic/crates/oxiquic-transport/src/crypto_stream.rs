//! CRYPTO-stream buffering for the TLS handshake (RFC 9000 Section 7.5, 19.6).
//!
//! Each packet-number space carries its own ordered byte stream of `CRYPTO`
//! frame data feeding rustls's `read_hs`. [`CryptoStream`] reassembles
//! out-of-order `CRYPTO` frames into the contiguous prefix that can be delivered
//! to TLS, and buffers outgoing handshake bytes produced by `write_hs` so they
//! can be chunked into `CRYPTO` frames as space permits.

use std::collections::{BTreeMap, VecDeque};

/// Maximum number of bytes of *out-of-order* (gap-blocked) CRYPTO data a single
/// packet-number space will buffer before raising `CRYPTO_BUFFER_EXCEEDED`
/// (RFC 9000 Section 7.5 / Section 20.1 code `0x0d`).
///
/// 64 KiB comfortably exceeds any realistic TLS 1.3 handshake flight (a large
/// certificate chain is a few tens of KiB) while bounding the state an
/// unauthenticated peer can pin: Initial keys are derived from the public
/// Destination Connection ID (RFC 9001 Section 5.2), so a spoofed Initial can
/// reach this buffer before any peer authentication has happened.
pub const MAX_CRYPTO_BUFFER_BYTES: usize = 64 * 1024;

/// Maximum number of distinct out-of-order segments buffered per space.
///
/// A byte budget alone is not sufficient: one-byte segments at scattered
/// offsets each cost a `BTreeMap` node far larger than their payload, so the
/// segment count is capped independently.
pub const MAX_CRYPTO_BUFFER_SEGMENTS: usize = 128;

/// The largest offset representable by the 62-bit varint encoding used for the
/// CRYPTO frame `Offset` and `Length` fields (RFC 9000 Section 16, Section 19.6).
const MAX_CRYPTO_OFFSET: u64 = (1u64 << 62) - 1;

/// Why a received `CRYPTO` frame was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CryptoStreamError {
    /// The peer sent more out-of-order CRYPTO data than this space will buffer.
    /// Maps to `CRYPTO_BUFFER_EXCEEDED` (RFC 9000 Section 20.1, `0x0d`).
    BufferExceeded,
    /// `offset + length` exceeds `2^62 - 1`, which RFC 9000 Section 19.6 makes a
    /// `FRAME_ENCODING_ERROR`.
    OffsetTooLarge,
}

/// A CRYPTO segment awaiting retransmission after its carrying packet was lost
/// (RFC 9002 Section 6.2): resent at its original offset.
#[derive(Debug, Clone)]
struct ResendSegment {
    offset: u64,
    data: Vec<u8>,
}

/// Ordered reassembly + send buffering for one space's CRYPTO stream.
#[derive(Debug, Default)]
pub struct CryptoStream {
    /// Next contiguous receive offset that has been delivered to TLS.
    recv_offset: u64,
    /// Buffered out-of-order received segments, keyed by start offset.
    recv_pending: BTreeMap<u64, Vec<u8>>,
    /// Outbound handshake bytes not yet sent in a CRYPTO frame.
    send_buf: Vec<u8>,
    /// Absolute offset of the first byte currently in `send_buf`.
    send_base: u64,
    /// Lost CRYPTO segments to retransmit ahead of fresh data (RFC 9002 6.2).
    resend: VecDeque<ResendSegment>,
}

impl CryptoStream {
    /// Create an empty CRYPTO stream.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of bytes currently held in the out-of-order gap buffer.
    ///
    /// Recomputed on demand; `recv_pending` is capped at
    /// [`MAX_CRYPTO_BUFFER_SEGMENTS`] entries so this is a bounded, cheap walk.
    fn pending_bytes(&self) -> usize {
        self.recv_pending.values().map(Vec::len).sum()
    }

    /// Accept a received `CRYPTO` frame at `offset`, returning the newly
    /// contiguous bytes (if any) to hand to `read_hs`.
    ///
    /// Duplicated or already-delivered data is dropped; gaps are buffered until
    /// the missing prefix arrives.
    ///
    /// # Errors
    ///
    /// * [`CryptoStreamError::OffsetTooLarge`] when `offset + data.len()`
    ///   exceeds the 62-bit varint maximum (RFC 9000 Section 19.6).
    /// * [`CryptoStreamError::BufferExceeded`] when accepting the segment would
    ///   push the out-of-order buffer past [`MAX_CRYPTO_BUFFER_BYTES`] or
    ///   [`MAX_CRYPTO_BUFFER_SEGMENTS`], or when the segment starts more than
    ///   [`MAX_CRYPTO_BUFFER_BYTES`] beyond the contiguous receive offset
    ///   (RFC 9000 Section 7.5).
    pub fn recv(&mut self, offset: u64, data: &[u8]) -> Result<Option<Vec<u8>>, CryptoStreamError> {
        let end = offset
            .checked_add(data.len() as u64)
            .ok_or(CryptoStreamError::OffsetTooLarge)?;
        if end > MAX_CRYPTO_OFFSET {
            return Err(CryptoStreamError::OffsetTooLarge);
        }
        if end <= self.recv_offset {
            // Entirely old data.
            return Ok(None);
        }
        // Trim any prefix we've already delivered.
        let (offset, data) = if offset < self.recv_offset {
            let skip = (self.recv_offset - offset) as usize;
            (self.recv_offset, &data[skip..])
        } else {
            (offset, data)
        };
        if !data.is_empty() {
            // RFC 9000 Section 7.5: bound the unauthenticated reassembly state.
            // A segment that starts beyond the buffer window can never be joined
            // to the contiguous prefix without first exceeding the window, so it
            // is refused outright rather than buffered.
            if offset
                > self
                    .recv_offset
                    .saturating_add(MAX_CRYPTO_BUFFER_BYTES as u64)
            {
                return Err(CryptoStreamError::BufferExceeded);
            }
            // Only a segment that leaves a gap is retained; a segment starting
            // exactly at `recv_offset` is drained immediately below.
            if offset > self.recv_offset {
                let replaced = self.recv_pending.get(&offset).map_or(0, Vec::len);
                if replaced == 0 && self.recv_pending.len() >= MAX_CRYPTO_BUFFER_SEGMENTS {
                    return Err(CryptoStreamError::BufferExceeded);
                }
                let projected = self
                    .pending_bytes()
                    .saturating_sub(replaced)
                    .saturating_add(data.len());
                if projected > MAX_CRYPTO_BUFFER_BYTES {
                    return Err(CryptoStreamError::BufferExceeded);
                }
            }
            self.recv_pending.insert(offset, data.to_vec());
        }

        // Pop contiguous segments starting at recv_offset.
        let mut delivered = Vec::new();
        while let Some((&start, _)) = self.recv_pending.range(..=self.recv_offset).next_back() {
            if start > self.recv_offset {
                break;
            }
            let segment = match self.recv_pending.remove(&start) {
                Some(seg) => seg,
                None => break,
            };
            let seg_end = start + segment.len() as u64;
            if seg_end <= self.recv_offset {
                continue;
            }
            let skip = (self.recv_offset - start) as usize;
            delivered.extend_from_slice(&segment[skip..]);
            self.recv_offset = seg_end;
        }
        if delivered.is_empty() {
            Ok(None)
        } else {
            Ok(Some(delivered))
        }
    }

    /// Queue outbound handshake bytes produced by `write_hs`.
    pub fn enqueue_send(&mut self, data: &[u8]) {
        self.send_buf.extend_from_slice(data);
    }

    /// Whether there are outbound CRYPTO bytes awaiting transmission, including
    /// segments queued for retransmission.
    #[must_use]
    pub fn has_send_data(&self) -> bool {
        !self.resend.is_empty() || !self.send_buf.is_empty()
    }

    /// Re-queue a lost CRYPTO segment for retransmission at `offset` (RFC 9002
    /// Section 6.2). Retransmitted data is emitted before fresh send data.
    pub fn requeue(&mut self, offset: u64, data: Vec<u8>) {
        self.resend.push_back(ResendSegment { offset, data });
    }

    /// Take up to `max` bytes of outbound data as the next CRYPTO frame body,
    /// returning `(offset, bytes)`. Retransmittable segments are emitted first
    /// at their original offset; otherwise fresh buffered data is chunked,
    /// advancing the send cursor.
    #[must_use]
    pub fn take_send(&mut self, max: usize) -> Option<(u64, Vec<u8>)> {
        if max == 0 {
            return None;
        }
        // Retransmit lost segments first, splitting if larger than `max`.
        if let Some(front) = self.resend.front_mut() {
            if front.data.len() <= max {
                let seg = self.resend.pop_front()?;
                return Some((seg.offset, seg.data));
            }
            let chunk: Vec<u8> = front.data.drain(..max).collect();
            let offset = front.offset;
            front.offset += max as u64;
            return Some((offset, chunk));
        }
        if self.send_buf.is_empty() {
            return None;
        }
        let take = max.min(self.send_buf.len());
        let chunk: Vec<u8> = self.send_buf.drain(..take).collect();
        let offset = self.send_base;
        self.send_base += take as u64;
        Some((offset, chunk))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_order_delivery() {
        let mut s = CryptoStream::new();
        assert_eq!(s.recv(0, b"hello"), Ok(Some(b"hello".to_vec())));
        assert_eq!(s.recv(5, b"world"), Ok(Some(b"world".to_vec())));
    }

    #[test]
    fn out_of_order_buffered_then_flushed() {
        let mut s = CryptoStream::new();
        assert_eq!(s.recv(5, b"world"), Ok(None));
        assert_eq!(s.recv(0, b"hello"), Ok(Some(b"helloworld".to_vec())));
    }

    #[test]
    fn duplicate_dropped() {
        let mut s = CryptoStream::new();
        assert_eq!(s.recv(0, b"hello"), Ok(Some(b"hello".to_vec())));
        assert_eq!(s.recv(0, b"hello"), Ok(None));
        assert_eq!(s.recv(2, b"llo"), Ok(None));
    }

    #[test]
    fn partial_overlap() {
        let mut s = CryptoStream::new();
        assert_eq!(s.recv(0, b"abc"), Ok(Some(b"abc".to_vec())));
        // Overlaps delivered prefix, extends past it.
        assert_eq!(s.recv(1, b"bcdef"), Ok(Some(b"def".to_vec())));
    }

    /// Regression (RFC 9000 Section 7.5): a peer that never fills the gap at
    /// offset 0 used to pin one `BTreeMap` entry per spoofed Initial packet
    /// forever. The segment cap now refuses the flood.
    #[test]
    fn segment_flood_is_rejected() {
        let mut s = CryptoStream::new();
        let payload = [0u8; 8];
        // Every segment leaves the gap at offset 0 unfilled.
        for i in 0..MAX_CRYPTO_BUFFER_SEGMENTS as u64 {
            let offset = 16 + i * 16;
            assert_eq!(s.recv(offset, &payload), Ok(None), "segment {i} buffered");
        }
        let overflow_offset = 16 + (MAX_CRYPTO_BUFFER_SEGMENTS as u64) * 16;
        assert_eq!(
            s.recv(overflow_offset, &payload),
            Err(CryptoStreamError::BufferExceeded)
        );
        assert_eq!(s.recv_pending.len(), MAX_CRYPTO_BUFFER_SEGMENTS);
    }

    /// Regression: a handful of large gap segments must not be able to pin more
    /// than [`MAX_CRYPTO_BUFFER_BYTES`] even while staying under the segment cap.
    #[test]
    fn byte_flood_is_rejected() {
        let mut s = CryptoStream::new();
        let chunk = vec![0u8; 1024];
        let mut buffered = 0usize;
        for i in 0..MAX_CRYPTO_BUFFER_SEGMENTS as u64 {
            // Offsets stay inside the window but never reach offset 0.
            let offset = 1 + i * 1024;
            match s.recv(offset, &chunk) {
                Ok(None) => buffered += chunk.len(),
                Err(CryptoStreamError::BufferExceeded) => break,
                other => panic!("unexpected result at segment {i}: {other:?}"),
            }
        }
        assert!(buffered <= MAX_CRYPTO_BUFFER_BYTES);
        assert!(s.pending_bytes() <= MAX_CRYPTO_BUFFER_BYTES);
    }

    /// Regression: a segment far beyond the contiguous receive offset can never
    /// become deliverable, so it is refused rather than buffered.
    #[test]
    fn far_offset_is_rejected() {
        let mut s = CryptoStream::new();
        assert_eq!(
            s.recv(0x4000_0000, b"far"),
            Err(CryptoStreamError::BufferExceeded)
        );
        assert!(s.recv_pending.is_empty());
    }

    /// RFC 9000 Section 19.6: `offset + length` beyond `2^62 - 1` is a
    /// FRAME_ENCODING_ERROR, never an arithmetic overflow.
    #[test]
    fn offset_past_varint_max_is_rejected() {
        let mut s = CryptoStream::new();
        assert_eq!(
            s.recv(MAX_CRYPTO_OFFSET, b"xy"),
            Err(CryptoStreamError::OffsetTooLarge)
        );
        assert_eq!(
            s.recv(u64::MAX, b"xy"),
            Err(CryptoStreamError::OffsetTooLarge)
        );
    }

    /// A legitimate handshake that arrives out of order but within the window
    /// still reassembles.
    #[test]
    fn large_in_window_reassembly_still_works() {
        let mut s = CryptoStream::new();
        let tail = vec![7u8; 4096];
        assert_eq!(s.recv(4096, &tail), Ok(None));
        let head = vec![3u8; 4096];
        let delivered = s.recv(0, &head).expect("gap filled").expect("bytes");
        assert_eq!(delivered.len(), 8192);
        assert!(s.recv_pending.is_empty());
    }

    #[test]
    fn requeue_drains_before_send_buf() {
        let mut s = CryptoStream::new();
        s.enqueue_send(b"fresh-crypto");
        s.requeue(50, b"lost-crypto".to_vec());
        assert!(s.has_send_data());
        let (off, data) = s.take_send(100).expect("resend first");
        assert_eq!((off, &data[..]), (50, &b"lost-crypto"[..]));
        let (off, data) = s.take_send(100).expect("then fresh");
        assert_eq!((off, &data[..]), (0, &b"fresh-crypto"[..]));
    }

    #[test]
    fn send_chunking() {
        let mut s = CryptoStream::new();
        s.enqueue_send(b"handshake-bytes");
        let (off, chunk) = s.take_send(9).expect("chunk");
        assert_eq!(off, 0);
        assert_eq!(chunk, b"handshake");
        let (off, chunk) = s.take_send(100).expect("chunk2");
        assert_eq!(off, 9);
        assert_eq!(chunk, b"-bytes");
        assert!(s.take_send(10).is_none());
    }
}
