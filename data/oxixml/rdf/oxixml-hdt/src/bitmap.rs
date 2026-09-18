//! `Bitmap` — a bit sequence with `rank`/`select` support, used to encode the parent/child
//! boundaries of a [`BitmapTriples`](crate::triples::BitmapTriples) adjacency structure.
//!
//! Per the specification, a `1` bit marks that the corresponding array element is the *last*
//! child of its parent group; `rank`/`select` over that bitmap is how a reader recovers, for
//! any parent index, the contiguous range of child indices that belong to it.
//!
//! # Wire format
//!
//! ```text
//! Byte    format         (1, the only bitmap format the specification defines)
//! VByte   number of bits
//! CRC8    of the two fields above
//! bytes   ceil(bits / 8) packed bits, LSB-first within each byte, "byte aligned, little endian"
//! CRC32C  of the packed bytes
//! ```
//!
//! # In-memory index
//!
//! The persisted form is just the raw bits — `rank`/`select` acceleration is rebuilt in memory
//! after loading (or while building), never stored in the file. This mirrors the task's
//! "simple rank9-like structure" scope: a two-level scheme with a cumulative popcount cached
//! every `BLOCK_BITS` bits (a *block*), falling back to a linear byte scan inside the block.
//! This is `O(len / BLOCK_BITS)` words of overhead and `O(BLOCK_BITS)` work per query — simple,
//! correct, and easily fast enough for HDT files sized for tests and small-to-medium corpora,
//! at the cost of the sub-block binary-search cleverness a production `rank9` implementation
//! would add.

use crate::error::{HdtError, Result};
use crate::framing::{read_metadata, read_payload, write_metadata, write_payload};

/// The bitmap format identifier (the only one the specification defines).
const FORMAT_PLAIN: u8 = 1;

/// The block size, in bits, at which cumulative rank counts are cached.
const BLOCK_BITS: usize = 512;

/// Builds a [`Bitmap`] bit-by-bit before its `rank`/`select` index is computed.
#[derive(Debug, Clone)]
pub struct BitmapBuilder {
    len: usize,
    data: Vec<u8>,
}

impl BitmapBuilder {
    /// Creates a builder for `len` bits, all initially `0`.
    #[must_use]
    pub fn zeroed(len: usize) -> Self {
        Self {
            len,
            data: vec![0; len.div_ceil(8)],
        }
    }

    /// Sets the bit at `index` to `1`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    pub fn set_one(&mut self, index: usize) {
        assert!(
            index < self.len,
            "bit index {index} out of bounds ({})",
            self.len
        );
        self.data[index / 8] |= 1 << (index % 8);
    }

    /// Finalizes the bitmap, computing its `rank`/`select` index.
    #[must_use]
    pub fn finish(self) -> Bitmap {
        Bitmap::from_raw(self.len, self.data)
    }
}

/// An immutable bit sequence supporting `O(BLOCK_BITS)` `rank1`/`select1` queries.
#[derive(Debug, Clone)]
pub struct Bitmap {
    len: usize,
    data: Vec<u8>,
    /// `block_rank[b]` is the number of `1` bits in `[0, b * BLOCK_BITS)`.
    block_rank: Vec<u64>,
    total_ones: u64,
}

impl Bitmap {
    /// Builds the `rank`/`select` index over already-packed bits.
    fn from_raw(len: usize, data: Vec<u8>) -> Self {
        let num_blocks = len.div_ceil(BLOCK_BITS);
        let mut block_rank = Vec::with_capacity(num_blocks + 1);
        let mut running: u64 = 0;
        block_rank.push(0);
        for block in 0..num_blocks {
            let start_bit = block * BLOCK_BITS;
            let end_bit = ((block + 1) * BLOCK_BITS).min(len);
            running += count_ones_in_range(&data, start_bit, end_bit);
            block_rank.push(running);
        }
        Self {
            len,
            data,
            block_rank,
            total_ones: running,
        }
    }

    /// Creates a bitmap of all zero bits.
    #[must_use]
    pub fn zeroed(len: usize) -> Self {
        BitmapBuilder::zeroed(len).finish()
    }

    /// The number of bits in this bitmap.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this bitmap has no bits.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The total number of `1` bits.
    #[must_use]
    #[inline]
    pub const fn count_ones(&self) -> u64 {
        self.total_ones
    }

    /// Reads the bit at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[must_use]
    pub fn get(&self, index: usize) -> bool {
        assert!(
            index < self.len,
            "bit index {index} out of bounds ({})",
            self.len
        );
        (self.data[index / 8] >> (index % 8)) & 1 == 1
    }

    /// Counts the `1` bits in `[0, index)`.
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    #[must_use]
    pub fn rank1(&self, index: usize) -> u64 {
        assert!(
            index <= self.len,
            "rank index {index} out of bounds ({})",
            self.len
        );
        let block = index / BLOCK_BITS;
        let base = self.block_rank[block];
        base + count_ones_in_range(&self.data, block * BLOCK_BITS, index)
    }

    /// Finds the `0`-based position of the `k`-th `1` bit, using `1`-based `k`.
    ///
    /// Returns `None` if `k == 0` or `k` exceeds [`count_ones`](Self::count_ones).
    #[must_use]
    pub fn select1(&self, k: u64) -> Option<usize> {
        if k == 0 || k > self.total_ones {
            return None;
        }
        // Binary search the block index: the largest block whose prefix rank is still < k.
        let mut low = 0usize;
        let mut high = self.block_rank.len() - 1;
        while low + 1 < high {
            let mid = low + (high - low) / 2;
            if self.block_rank[mid] < k {
                low = mid;
            } else {
                high = mid;
            }
        }
        let block = low;
        let mut remaining = k - self.block_rank[block];
        let start_bit = block * BLOCK_BITS;
        let end_bit = self.len.min(start_bit + BLOCK_BITS);
        for bit in start_bit..end_bit {
            if self.get(bit) {
                remaining -= 1;
                if remaining == 0 {
                    return Some(bit);
                }
            }
        }
        None
    }

    /// Serializes this bitmap using the framed wire format, appending to `out`.
    pub fn write(&self, out: &mut Vec<u8>) {
        write_metadata(out, FORMAT_PLAIN, &[self.len as u64]);
        write_payload(out, &self.data);
    }

    /// Deserializes a [`Bitmap`] previously written by [`write`](Self::write), starting at
    /// `*pos` and advancing it past the bitmap.
    ///
    /// # Errors
    ///
    /// Returns an [`HdtError`] if the framing is truncated, a checksum fails, or the bitmap
    /// declares an unsupported format.
    pub fn read(data: &[u8], pos: &mut usize, section: &'static str) -> Result<Self> {
        let (format, fields) = read_metadata(data, pos, section, 1)?;
        if format != FORMAT_PLAIN {
            return Err(HdtError::UnsupportedFormat {
                section,
                found: format!("bitmap format {format}"),
            });
        }
        let len = fields[0] as usize;
        let byte_len = len.div_ceil(8);
        let payload = read_payload(data, pos, section, byte_len)?;
        Ok(Self::from_raw(len, payload.to_vec()))
    }
}

impl Default for Bitmap {
    /// The empty bitmap (`0` bits).
    #[inline]
    fn default() -> Self {
        Self::zeroed(0)
    }
}

/// Counts the `1` bits in `data` over the half-open bit range `[start_bit, end_bit)`.
fn count_ones_in_range(data: &[u8], start_bit: usize, end_bit: usize) -> u64 {
    if start_bit >= end_bit {
        return 0;
    }
    let mut count: u64 = 0;
    for bit in start_bit..end_bit {
        if (data[bit / 8] >> (bit % 8)) & 1 == 1 {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Bitmap {
        // Bits set at 0, 3, 4, 10, 511, 512, 1000 out of 1200.
        let mut builder = BitmapBuilder::zeroed(1200);
        for index in [0, 3, 4, 10, 511, 512, 1000] {
            builder.set_one(index);
        }
        builder.finish()
    }

    #[test]
    fn get_reflects_set_bits() {
        let bitmap = sample();
        assert!(bitmap.get(0));
        assert!(!bitmap.get(1));
        assert!(bitmap.get(511));
        assert!(bitmap.get(512));
        assert!(bitmap.get(1000));
        assert!(!bitmap.get(1199));
    }

    #[test]
    fn count_ones_matches() {
        assert_eq!(sample().count_ones(), 7);
    }

    #[test]
    fn rank1_matches_naive() {
        let bitmap = sample();
        let ones = [0usize, 3, 4, 10, 511, 512, 1000];
        for index in 0..=1200 {
            let expected = ones.iter().filter(|&&one| one < index).count() as u64;
            assert_eq!(bitmap.rank1(index), expected, "rank1({index})");
        }
    }

    #[test]
    fn select1_matches_naive() {
        let bitmap = sample();
        let ones = [0usize, 3, 4, 10, 511, 512, 1000];
        for (k, &expected) in ones.iter().enumerate() {
            assert_eq!(bitmap.select1(k as u64 + 1), Some(expected));
        }
        assert_eq!(bitmap.select1(0), None);
        assert_eq!(bitmap.select1(8), None);
    }

    #[test]
    fn wire_roundtrip() -> Result<()> {
        let bitmap = sample();
        let mut buffer = Vec::new();
        bitmap.write(&mut buffer);
        let mut pos = 0;
        let decoded = Bitmap::read(&buffer, &mut pos, "test")?;
        assert_eq!(pos, buffer.len());
        assert_eq!(decoded.len(), bitmap.len());
        assert_eq!(decoded.count_ones(), bitmap.count_ones());
        for index in 0..bitmap.len() {
            assert_eq!(decoded.get(index), bitmap.get(index));
        }
        Ok(())
    }

    #[test]
    fn wire_roundtrip_empty() -> Result<()> {
        let bitmap = Bitmap::zeroed(0);
        let mut buffer = Vec::new();
        bitmap.write(&mut buffer);
        let mut pos = 0;
        let decoded = Bitmap::read(&buffer, &mut pos, "test")?;
        assert!(decoded.is_empty());
        assert_eq!(decoded.select1(1), None);
        Ok(())
    }

    #[test]
    fn all_zero_bitmap() {
        let bitmap = Bitmap::zeroed(100);
        assert_eq!(bitmap.count_ones(), 0);
        assert_eq!(bitmap.rank1(50), 0);
        assert_eq!(bitmap.select1(1), None);
    }

    #[test]
    fn all_one_bitmap() {
        let mut builder = BitmapBuilder::zeroed(64);
        for index in 0..64 {
            builder.set_one(index);
        }
        let bitmap = builder.finish();
        assert_eq!(bitmap.count_ones(), 64);
        assert_eq!(bitmap.rank1(64), 64);
        assert_eq!(bitmap.select1(64), Some(63));
    }

    #[test]
    fn rejects_unsupported_format() {
        let mut buffer = Vec::new();
        write_metadata(&mut buffer, 2, &[0]);
        write_payload(&mut buffer, &[]);
        let mut pos = 0;
        let error = Bitmap::read(&buffer, &mut pos, "test").expect_err("expected an error");
        assert!(matches!(error, HdtError::UnsupportedFormat { .. }));
    }
}
