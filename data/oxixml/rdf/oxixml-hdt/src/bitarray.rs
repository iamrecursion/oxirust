//! `LogArray` — a bit-packed array of fixed-width unsigned integers.
//!
//! HDT never stores IDs and offsets as full 32/64-bit words: every array of integers (PFC block
//! pointers, `BitmapTriples` adjacency arrays, ...) is packed using the *minimum* number of bits
//! that can represent its largest element, per the specification's `Log64` array format.
//!
//! # Wire format
//!
//! ```text
//! Byte    format               (1 = Log64, the only variant this crate writes)
//! VByte   entry count
//! VByte   bits per entry
//! CRC8    of the three fields above
//! bytes   ceil(count * bits_per_entry / 8) packed entries
//! CRC32C  of the packed bytes
//! ```
//!
//! Entries are packed **LSB-first**: entry `i` occupies bits `[i * bits, i * bits + bits)` of a
//! virtual little-endian bit stream (bit `b` of that stream lives in byte `b / 8`, bit position
//! `b % 8`, counting from the least-significant bit of each byte). The specification's prose
//! does not spell out a bit-packing byte order for `Log64`, so this crate documents and commits
//! to this one — it is internally consistent for round-tripping and matches the "byte aligned,
//! little endian" wording used for the sibling `Bitmap` format (see [`crate::bitmap`]).
//!
//! The `bits per entry` field is written explicitly (rather than left implicit) so a `LogArray`
//! can always be decoded on its own, without needing side-channel knowledge of the maximum
//! value it stores.

use crate::error::{HdtError, Result};
use crate::framing::{read_metadata, read_payload, write_metadata, write_payload};

/// The `Log64` array-format identifier.
const FORMAT_LOG64: u8 = 1;

/// A bit-packed array of `u64` values, each stored in a fixed number of bits.
#[derive(Debug, Clone, Default)]
pub struct BitArray {
    bits_per_entry: u32,
    len: usize,
    data: Vec<u8>,
}

impl BitArray {
    /// Creates a zero-filled array of `len` entries, each `bits_per_entry` bits wide.
    ///
    /// `bits_per_entry` is clamped to be at least `1` (a zero-width array cannot address its
    /// own entries) and at most `64`.
    #[must_use]
    pub fn zeroed(len: usize, bits_per_entry: u32) -> Self {
        let bits_per_entry = bits_per_entry.clamp(1, 64);
        let total_bits = len.saturating_mul(bits_per_entry as usize);
        let byte_len = total_bits.div_ceil(8);
        Self {
            bits_per_entry,
            len,
            data: vec![0; byte_len],
        }
    }

    /// Packs `values` into the narrowest [`BitArray`] that can hold all of them.
    ///
    /// The width is derived from the largest value present (minimum `1` bit, so an all-zero
    /// array is still representable and decodable).
    #[must_use]
    pub fn from_values(values: &[u64]) -> Self {
        let max = values.iter().copied().max().unwrap_or(0);
        let bits = bits_needed(max);
        let mut array = Self::zeroed(values.len(), bits);
        for (index, &value) in values.iter().enumerate() {
            array.set(index, value);
        }
        array
    }

    /// The number of entries stored.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this array has no entries.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of bits used to store each entry.
    #[must_use]
    #[inline]
    pub const fn bits_per_entry(&self) -> u32 {
        self.bits_per_entry
    }

    /// Reads the entry at `index`, or `None` if `index` is out of bounds.
    ///
    /// Prefer this over [`get`](Self::get) whenever `index` is derived from untrusted file
    /// contents: a malformed HDT file must yield an
    /// [`HdtError`], never a panic.
    #[must_use]
    pub fn try_get(&self, index: usize) -> Option<u64> {
        (index < self.len).then(|| self.get_unchecked_index(index))
    }

    /// Reads the entry at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[must_use]
    pub fn get(&self, index: usize) -> u64 {
        assert!(
            index < self.len,
            "BitArray index {index} out of bounds ({})",
            self.len
        );
        self.get_unchecked_index(index)
    }

    /// Reads the entry at `index` assuming it has already been bounds-checked.
    ///
    /// Still memory-safe: `index >= len` simply reads whatever bits the backing storage happens
    /// to hold (or `0` past its end).
    fn get_unchecked_index(&self, index: usize) -> u64 {
        let bits = self.bits_per_entry as usize;
        let start_bit = index * bits;
        let mut value: u64 = 0;
        for bit_offset in 0..bits {
            let bit_index = start_bit + bit_offset;
            let byte = self.data.get(bit_index / 8).copied().unwrap_or(0);
            let bit = (byte >> (bit_index % 8)) & 1;
            value |= u64::from(bit) << bit_offset;
        }
        value
    }

    /// Overwrites the entry at `index` with `value`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()` or if `value` does not fit in `bits_per_entry()` bits.
    pub fn set(&mut self, index: usize, value: u64) {
        assert!(
            index < self.len,
            "BitArray index {index} out of bounds ({})",
            self.len
        );
        let bits = self.bits_per_entry as usize;
        assert!(
            bits == 64 || value < (1u64 << bits),
            "value {value} does not fit in {bits} bits"
        );
        let start_bit = index * bits;
        for bit_offset in 0..bits {
            let bit_index = start_bit + bit_offset;
            let byte_index = bit_index / 8;
            let mask = 1u8 << (bit_index % 8);
            if (value >> bit_offset) & 1 == 1 {
                self.data[byte_index] |= mask;
            } else {
                self.data[byte_index] &= !mask;
            }
        }
    }

    /// Iterates over all entries in order.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.len).map(move |index| self.get(index))
    }

    /// Serializes this array using the framed `Log64` wire format, appending to `out`.
    pub fn write(&self, out: &mut Vec<u8>) {
        write_metadata(
            out,
            FORMAT_LOG64,
            &[self.len as u64, u64::from(self.bits_per_entry)],
        );
        write_payload(out, &self.data);
    }

    /// Deserializes a [`BitArray`] previously written by [`write`](Self::write), starting at
    /// `*pos` and advancing it past the array.
    ///
    /// # Errors
    ///
    /// Returns an [`HdtError`] if the framing is truncated, a checksum fails, or the array
    /// declares an unsupported format.
    pub fn read(data: &[u8], pos: &mut usize, section: &'static str) -> Result<Self> {
        let (format, fields) = read_metadata(data, pos, section, 2)?;
        if format != FORMAT_LOG64 {
            return Err(HdtError::UnsupportedFormat {
                section,
                found: format!("array format {format}"),
            });
        }
        let len = fields[0] as usize;
        let bits_per_entry = u32::try_from(fields[1]).map_err(|_| {
            HdtError::Malformed(format!(
                "{section}: bits-per-entry {} does not fit in a u32",
                fields[1]
            ))
        })?;
        let bits_per_entry = bits_per_entry.clamp(1, 64);
        let total_bits = len.saturating_mul(bits_per_entry as usize);
        let byte_len = total_bits.div_ceil(8);
        let payload = read_payload(data, pos, section, byte_len)?;
        Ok(Self {
            bits_per_entry,
            len,
            data: payload.to_vec(),
        })
    }
}

/// The minimum number of bits needed to represent every value in `0..=max` (at least `1`).
#[must_use]
pub const fn bits_needed(max: u64) -> u32 {
    if max == 0 {
        1
    } else {
        64 - max.leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_set_roundtrip() {
        let mut array = BitArray::zeroed(10, 5);
        for i in 0..10 {
            array.set(i, (i as u64 * 7) % 32);
        }
        for i in 0..10 {
            assert_eq!(array.get(i), (i as u64 * 7) % 32);
        }
    }

    #[test]
    fn from_values_picks_minimal_width() {
        let array = BitArray::from_values(&[0, 1, 2, 3]);
        assert_eq!(array.bits_per_entry(), 2);
        let array = BitArray::from_values(&[0]);
        assert_eq!(array.bits_per_entry(), 1);
        let array = BitArray::from_values(&[300]);
        assert_eq!(array.bits_per_entry(), 9);
    }

    #[test]
    fn bits_needed_edge_cases() {
        assert_eq!(bits_needed(0), 1);
        assert_eq!(bits_needed(1), 1);
        assert_eq!(bits_needed(2), 2);
        assert_eq!(bits_needed(3), 2);
        assert_eq!(bits_needed(4), 3);
        assert_eq!(bits_needed(u64::MAX), 64);
    }

    #[test]
    fn wire_roundtrip() -> Result<()> {
        let values: Vec<u64> = (0..500).map(|i| (i * 31) % 1000).collect();
        let array = BitArray::from_values(&values);
        let mut buffer = Vec::new();
        array.write(&mut buffer);
        let mut pos = 0;
        let decoded = BitArray::read(&buffer, &mut pos, "test")?;
        assert_eq!(pos, buffer.len());
        assert_eq!(decoded.len(), values.len());
        assert_eq!(decoded.iter().collect::<Vec<_>>(), values);
        Ok(())
    }

    #[test]
    fn wire_roundtrip_empty() -> Result<()> {
        let array = BitArray::from_values(&[]);
        let mut buffer = Vec::new();
        array.write(&mut buffer);
        let mut pos = 0;
        let decoded = BitArray::read(&buffer, &mut pos, "test")?;
        assert!(decoded.is_empty());
        Ok(())
    }

    #[test]
    fn wire_roundtrip_full_width() -> Result<()> {
        let values = vec![u64::MAX, 0, u64::MAX / 2];
        let array = BitArray::from_values(&values);
        assert_eq!(array.bits_per_entry(), 64);
        let mut buffer = Vec::new();
        array.write(&mut buffer);
        let mut pos = 0;
        let decoded = BitArray::read(&buffer, &mut pos, "test")?;
        assert_eq!(decoded.iter().collect::<Vec<_>>(), values);
        Ok(())
    }

    #[test]
    fn rejects_unsupported_format() {
        let mut buffer = Vec::new();
        write_metadata(&mut buffer, 2, &[0, 1]);
        write_payload(&mut buffer, &[]);
        let mut pos = 0;
        let error = BitArray::read(&buffer, &mut pos, "test").expect_err("expected an error");
        assert!(matches!(error, HdtError::UnsupportedFormat { .. }));
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn set_rejects_oversized_value() {
        let mut array = BitArray::zeroed(1, 3);
        array.set(0, 8);
    }
}
