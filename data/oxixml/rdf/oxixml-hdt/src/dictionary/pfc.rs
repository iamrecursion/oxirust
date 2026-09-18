//! `DictionarySectionPlainFrontCoding` — a single, block-compressed string↔ID section.
//!
//! One [`PlainFrontCoding`] section maps a set of distinct, lexicographically **sorted**
//! strings to consecutive `1`-based IDs. Strings are grouped into fixed-size blocks; the first
//! string of each block is stored verbatim, and every later string in the block is stored as a
//! `(shared-prefix length, suffix)` pair relative to the *previous* string in the block (classic
//! front coding / incremental encoding). A [`crate::bitarray::BitArray`] of per-block byte
//! offsets lets lookups seek directly to the right block instead of decoding from the start of
//! the buffer.
//!
//! # Wire format
//!
//! ```text
//! u32 LE      section type = 2 (PlainFrontCoding)
//! VByte       total number of strings
//! VByte       length of the packed string buffer, in bytes
//! VByte       number of strings per front-coded block
//! CRC8        of the three VByte fields above
//! BitArray    per-block starting byte offset into the buffer (see crate::bitarray)
//! bytes       the packed buffer:
//!               <first string><NUL>
//!               (<VByte common-prefix length><suffix bytes><NUL>) * (block_size - 1), repeated
//!               per block; the last block may be shorter.
//! CRC32C      of the packed buffer
//! ```
//!
//! Every string (including each block's first, verbatim string) is NUL-terminated. All work is
//! performed on raw bytes; strings are only re-validated as UTF-8 at the very end of decoding,
//! so a shared prefix boundary never needs to land on a UTF-8 character boundary mid-algorithm.

use std::cmp::Ordering;

use crate::crc::crc8;
use crate::error::{HdtError, Result};
use crate::framing::{read_payload, write_payload};
use crate::vbyte::{read_vbyte, write_vbyte};

use crate::bitarray::BitArray;

/// The `DictionarySectionPlainFrontCoding` type identifier.
const SECTION_TYPE_PFC: u32 = 2;

/// The number of strings grouped into each front-coded block, as written by this crate.
///
/// Smaller blocks front-code less aggressively but make random-access lookups (`id_to_string`,
/// `string_to_id`) cheaper; the specification leaves the exact value implementation-defined
/// (its own worked example uses `8`). `16` is a reasonable middle ground.
///
/// The value actually in force for a section is *not* this constant but the one stored in the
/// section's own preamble (the `block_size` field), so sections produced by another
/// HDT implementation with a different block size decode correctly.
pub const BLOCK_SIZE: usize = 16;

/// The largest block size this crate accepts from a file, as a sanity bound on a field that is
/// otherwise attacker-controlled.
const MAX_BLOCK_SIZE: usize = 1 << 20;

/// One `DictionarySectionPlainFrontCoding` block: sorted strings, front-coded, ID-addressable.
#[derive(Debug, Clone)]
pub struct PlainFrontCoding {
    num_strings: usize,
    /// Strings per front-coded block; always `>= 1`, and read back from the file rather than
    /// assumed, so a section written with a different block size still decodes correctly.
    block_size: usize,
    block_pointers: BitArray,
    buffer: Vec<u8>,
}

impl Default for PlainFrontCoding {
    fn default() -> Self {
        Self {
            num_strings: 0,
            block_size: BLOCK_SIZE,
            block_pointers: BitArray::default(),
            buffer: Vec::new(),
        }
    }
}

impl PlainFrontCoding {
    /// Builds a section from `strings`, which **must** already be sorted in ascending byte
    /// order and contain no duplicates (the caller — [`crate::dictionary::FourSectionDictionary`]
    /// — is responsible for both).
    #[must_use]
    pub fn from_sorted_strings(strings: &[String]) -> Self {
        Self::from_sorted_strings_with_block_size(strings, BLOCK_SIZE)
    }

    /// Same as [`from_sorted_strings`](Self::from_sorted_strings) but with an explicit block size,
    /// which is recorded in the section so that readers do not have to guess it.
    ///
    /// A `block_size` of `0` is meaningless and is treated as `1`.
    #[must_use]
    pub fn from_sorted_strings_with_block_size(strings: &[String], block_size: usize) -> Self {
        let block_size = block_size.clamp(1, MAX_BLOCK_SIZE);
        let mut buffer = Vec::new();
        let mut block_pointers = Vec::with_capacity(strings.len().div_ceil(block_size));
        let mut previous: &[u8] = b"";
        for (index, string) in strings.iter().enumerate() {
            let bytes = string.as_bytes();
            if index % block_size == 0 {
                block_pointers.push(buffer.len() as u64);
                buffer.extend_from_slice(bytes);
            } else {
                let common = common_prefix_len(previous, bytes);
                write_vbyte(&mut buffer, common as u64);
                buffer.extend_from_slice(&bytes[common..]);
            }
            buffer.push(0);
            previous = bytes;
        }
        Self {
            num_strings: strings.len(),
            block_size,
            block_pointers: BitArray::from_values(&block_pointers),
            buffer,
        }
    }

    /// The number of strings stored in this section.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.num_strings
    }

    /// Returns `true` if this section stores no strings.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.num_strings == 0
    }

    /// Resolves the `1`-based `id` to its string.
    ///
    /// # Errors
    ///
    /// Returns [`HdtError::IdOutOfRange`] if `id` is `0` or greater than [`len`](Self::len), or
    /// [`HdtError::Malformed`] if the reconstructed bytes are not valid UTF-8.
    pub fn id_to_string(&self, id: u32) -> Result<String> {
        let bytes = self.id_to_bytes(id)?;
        String::from_utf8(bytes).map_err(|error| {
            HdtError::Malformed(format!("dictionary string is not valid UTF-8: {error}"))
        })
    }

    /// Resolves `string` to its `1`-based ID, if present.
    #[must_use]
    pub fn string_to_id(&self, string: &str) -> Option<u32> {
        let target = string.as_bytes();
        // Never trust `block_pointers.len()` alone: a file may store more pointers than
        // `num_strings` accounts for (a trailing sentinel, or plain corruption).
        let num_blocks = self
            .block_pointers
            .len()
            .min(self.num_strings.div_ceil(self.block_size));
        if num_blocks == 0 {
            return None;
        }

        // Binary search for the last block whose first string is <= target
        // (an "upper bound minus one" search over decoded block heads).
        let mut low = 0usize;
        let mut high = num_blocks;
        while low < high {
            let mid = low + (high - low) / 2;
            let head = self.block_head_bytes(mid);
            if head.as_slice() <= target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if low == 0 {
            return None;
        }
        let block = low - 1;

        let block_start_id = block * self.block_size;
        let count_in_block = self.block_size.min(self.num_strings - block_start_id);
        let mut pos = self.block_pointers.try_get(block)? as usize;
        let mut current = read_cstring_bytes(&self.buffer, &mut pos).ok()?;
        for offset in 0..count_in_block {
            if offset > 0 {
                let prefix_len = read_vbyte(&self.buffer, &mut pos).ok()? as usize;
                let suffix = read_cstring_bytes(&self.buffer, &mut pos).ok()?;
                current = chain(&current, prefix_len, &suffix);
            }
            match current.as_slice().cmp(target) {
                Ordering::Equal => return Some((block_start_id + offset + 1) as u32),
                Ordering::Greater => return None,
                Ordering::Less => {}
            }
        }
        None
    }

    /// Iterates over every string in ascending ID order (`1..=len`).
    #[must_use]
    pub fn iter(&self) -> PfcIter<'_> {
        PfcIter {
            section: self,
            pos: 0,
            current: Vec::new(),
            next_index: 0,
        }
    }

    /// Serializes this section using its framed wire format, appending to `out`.
    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&SECTION_TYPE_PFC.to_le_bytes());
        let start = out.len();
        write_vbyte(out, self.num_strings as u64);
        write_vbyte(out, self.buffer.len() as u64);
        write_vbyte(out, self.block_size as u64);
        out.push(crc8(&out[start..]));
        self.block_pointers.write(out);
        write_payload(out, &self.buffer);
    }

    /// Deserializes a section previously written by [`write`](Self::write), starting at
    /// `*pos` and advancing it past the section.
    ///
    /// # Errors
    ///
    /// Returns an [`HdtError`] if the framing is truncated, a checksum fails, or the section
    /// declares an unsupported type.
    pub fn read(data: &[u8], pos: &mut usize, section: &'static str) -> Result<Self> {
        let type_bytes = data
            .get(*pos..*pos + 4)
            .ok_or(HdtError::UnexpectedEof { section })?;
        let section_type = u32::from_le_bytes(
            type_bytes
                .try_into()
                .expect("slice of length 4 converts into [u8; 4]"),
        );
        *pos += 4;
        if section_type != SECTION_TYPE_PFC {
            return Err(HdtError::UnsupportedFormat {
                section,
                found: format!("dictionary section type {section_type}"),
            });
        }

        let start = *pos;
        let num_strings = read_vbyte(data, pos)? as usize;
        let buffer_len = read_vbyte(data, pos)? as usize;
        let block_size = read_vbyte(data, pos)? as usize;
        let end = *pos;
        let computed = crc8(&data[start..end]);
        let stored = *data.get(*pos).ok_or(HdtError::UnexpectedEof { section })?;
        *pos += 1;
        if stored != computed {
            return Err(HdtError::CrcMismatch {
                kind: "CRC8",
                section,
                expected: u32::from(stored),
                actual: u32::from(computed),
            });
        }

        if block_size == 0 || block_size > MAX_BLOCK_SIZE {
            return Err(HdtError::Malformed(format!(
                "dictionary section {section} declares an unusable front-coding block size of {block_size}"
            )));
        }

        let block_pointers = BitArray::read(data, pos, section)?;
        // Cross-check the two independently-read fields against each other, the way
        // `BitmapTriples::read` does: a section that declares more strings than it stored block
        // pointers for would otherwise make a later ID lookup index `block_pointers` out of
        // bounds. (Extra pointers are tolerated: some writers append an end-of-data sentinel.)
        let required_blocks = num_strings.div_ceil(block_size);
        if block_pointers.len() < required_blocks {
            return Err(HdtError::Malformed(format!(
                "dictionary section {section} declares {num_strings} strings, which need \
                 {required_blocks} block pointers, but only {} are stored",
                block_pointers.len()
            )));
        }

        let buffer = read_payload(data, pos, section, buffer_len)?.to_vec();

        Ok(Self {
            num_strings,
            block_size,
            block_pointers,
            buffer,
        })
    }

    /// Decodes just the (verbatim, first) string of block `block`.
    fn block_head_bytes(&self, block: usize) -> Vec<u8> {
        let Some(offset) = self.block_pointers.try_get(block) else {
            return Vec::new();
        };
        let mut pos = offset as usize;
        read_cstring_bytes(&self.buffer, &mut pos).unwrap_or_default()
    }

    /// Resolves the `1`-based `id` to its raw bytes.
    fn id_to_bytes(&self, id: u32) -> Result<Vec<u8>> {
        if id == 0 || id as usize > self.num_strings {
            return Err(HdtError::IdOutOfRange {
                section: "dictionary",
                id: u64::from(id),
                len: self.num_strings as u64,
            });
        }
        let index0 = (id - 1) as usize;
        let block = index0 / self.block_size;
        let within = index0 % self.block_size;
        // `try_get`, not `get`: `num_strings` comes from the file and may disagree with the
        // stored block pointers, which must be an `HdtError` rather than a panic.
        let mut pos = self.block_pointers.try_get(block).ok_or_else(|| {
            HdtError::Malformed(format!(
                "dictionary section has no block pointer {block} for id {id}"
            ))
        })? as usize;
        let mut current = read_cstring_bytes(&self.buffer, &mut pos)?;
        for _ in 0..within {
            let prefix_len = read_vbyte(&self.buffer, &mut pos)? as usize;
            let suffix = read_cstring_bytes(&self.buffer, &mut pos)?;
            current = chain(&current, prefix_len, &suffix);
        }
        Ok(current)
    }
}

/// Reconstructs a front-coded string from its predecessor's bytes, a shared-prefix length, and
/// this string's own suffix bytes.
fn chain(previous: &[u8], prefix_len: usize, suffix: &[u8]) -> Vec<u8> {
    let mut value = Vec::with_capacity(prefix_len + suffix.len());
    value.extend_from_slice(&previous[..prefix_len.min(previous.len())]);
    value.extend_from_slice(suffix);
    value
}

/// The number of leading bytes `a` and `b` have in common.
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Reads a NUL-terminated byte string starting at `*pos`, advancing it past the terminator.
fn read_cstring_bytes(data: &[u8], pos: &mut usize) -> Result<Vec<u8>> {
    let start = *pos;
    let mut end = start;
    loop {
        let byte = *data.get(end).ok_or(HdtError::UnexpectedEof {
            section: "dictionary",
        })?;
        if byte == 0 {
            break;
        }
        end += 1;
    }
    let bytes = data[start..end].to_vec();
    *pos = end + 1;
    Ok(bytes)
}

/// Iterates over every string of a [`PlainFrontCoding`] section in ascending ID order.
#[derive(Debug)]
pub struct PfcIter<'a> {
    section: &'a PlainFrontCoding,
    pos: usize,
    current: Vec<u8>,
    next_index: usize,
}

impl Iterator for PfcIter<'_> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.section.num_strings {
            return None;
        }
        if self.next_index % self.section.block_size == 0 {
            self.current = read_cstring_bytes(&self.section.buffer, &mut self.pos).ok()?;
        } else {
            let prefix_len = read_vbyte(&self.section.buffer, &mut self.pos).ok()? as usize;
            let suffix = read_cstring_bytes(&self.section.buffer, &mut self.pos).ok()?;
            self.current = chain(&self.current, prefix_len, &suffix);
        }
        self.next_index += 1;
        String::from_utf8(self.current.clone()).ok()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.section.num_strings - self.next_index;
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_strings() -> Vec<String> {
        let mut strings: Vec<String> = vec![
            "alpha",
            "alphabet",
            "alps",
            "beta",
            "beta2",
            "gamma",
            "épée",
            "日本語",
            "日本語です",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        strings.sort();
        strings.dedup();
        strings
    }

    #[test]
    fn id_to_string_and_back() {
        let strings = sample_strings();
        let pfc = PlainFrontCoding::from_sorted_strings(&strings);
        assert_eq!(pfc.len(), strings.len());
        for (index, expected) in strings.iter().enumerate() {
            let id = (index + 1) as u32;
            assert_eq!(&pfc.id_to_string(id).expect("id in range"), expected);
            assert_eq!(pfc.string_to_id(expected), Some(id));
        }
    }

    #[test]
    fn string_to_id_missing_returns_none() {
        let strings = sample_strings();
        let pfc = PlainFrontCoding::from_sorted_strings(&strings);
        assert_eq!(pfc.string_to_id("not-present"), None);
        assert_eq!(pfc.string_to_id(""), None);
        assert_eq!(pfc.string_to_id("zzzzz"), None);
    }

    #[test]
    fn id_to_string_out_of_range() {
        let strings = sample_strings();
        let pfc = PlainFrontCoding::from_sorted_strings(&strings);
        assert!(matches!(
            pfc.id_to_string(0).expect_err("expected an error"),
            HdtError::IdOutOfRange { .. }
        ));
        assert!(matches!(
            pfc.id_to_string((strings.len() + 1) as u32)
                .expect_err("expected an error"),
            HdtError::IdOutOfRange { .. }
        ));
    }

    #[test]
    fn iter_yields_all_strings_in_order() {
        let strings = sample_strings();
        let pfc = PlainFrontCoding::from_sorted_strings(&strings);
        let collected: Vec<String> = pfc.iter().collect();
        assert_eq!(collected, strings);
    }

    #[test]
    fn spans_multiple_blocks() {
        let strings: Vec<String> = (0..500).map(|i| format!("item-{i:05}")).collect();
        let pfc = PlainFrontCoding::from_sorted_strings(&strings);
        for (index, expected) in strings.iter().enumerate() {
            let id = (index + 1) as u32;
            assert_eq!(&pfc.id_to_string(id).expect("id in range"), expected);
        }
        assert_eq!(pfc.string_to_id("item-00250"), Some(251));
    }

    #[test]
    fn empty_section() {
        let pfc = PlainFrontCoding::from_sorted_strings(&[]);
        assert!(pfc.is_empty());
        assert_eq!(pfc.string_to_id("anything"), None);
        assert_eq!(pfc.iter().count(), 0);
    }

    #[test]
    fn wire_roundtrip() -> Result<()> {
        let strings = sample_strings();
        let pfc = PlainFrontCoding::from_sorted_strings(&strings);
        let mut buffer = Vec::new();
        pfc.write(&mut buffer);
        let mut pos = 0;
        let decoded = PlainFrontCoding::read(&buffer, &mut pos, "test")?;
        assert_eq!(pos, buffer.len());
        assert_eq!(decoded.iter().collect::<Vec<_>>(), strings);
        Ok(())
    }

    #[test]
    fn wire_roundtrip_empty() -> Result<()> {
        let pfc = PlainFrontCoding::from_sorted_strings(&[]);
        let mut buffer = Vec::new();
        pfc.write(&mut buffer);
        let mut pos = 0;
        let decoded = PlainFrontCoding::read(&buffer, &mut pos, "test")?;
        assert!(decoded.is_empty());
        Ok(())
    }

    /// Regression test: a section written with a *different* block size (as another, equally
    /// spec-conforming HDT implementation may do — the specification leaves the value
    /// implementation-defined) must decode to exactly the same strings.
    #[test]
    fn decodes_sections_written_with_another_block_size() -> Result<()> {
        let strings: Vec<String> = (0..40).map(|i| format!("p{i:02}")).collect();
        for block_size in [1usize, 2, 3, 8, 16, 64, 4096] {
            let pfc = PlainFrontCoding::from_sorted_strings_with_block_size(&strings, block_size);
            let mut buffer = Vec::new();
            pfc.write(&mut buffer);
            let mut pos = 0;
            let decoded = PlainFrontCoding::read(&buffer, &mut pos, "test")?;
            assert_eq!(decoded.block_size, block_size);
            assert_eq!(decoded.iter().collect::<Vec<_>>(), strings);
            for (index, expected) in strings.iter().enumerate() {
                let id = (index + 1) as u32;
                assert_eq!(
                    &decoded.id_to_string(id)?,
                    expected,
                    "block_size {block_size}"
                );
                assert_eq!(decoded.string_to_id(expected), Some(id));
            }
            assert_eq!(decoded.string_to_id("missing"), None);
        }
        Ok(())
    }

    /// Builds a valid one-string section and rewrites its first VByte preamble field, fixing up
    /// the covering CRC8 so that the crafted file still verifies.
    fn section_with_patched_preamble(field: usize, value: u64) -> Vec<u8> {
        let pfc = PlainFrontCoding::from_sorted_strings(&["a".to_owned()]);
        let mut buffer = Vec::new();
        pfc.write(&mut buffer);
        // 4-byte section type, then three single-byte VBytes, then the CRC8 covering them.
        assert!(value <= 0x7F, "test only patches single-byte VBytes");
        buffer[4 + field] = (value as u8) | 0x80;
        buffer[7] = crc8(&buffer[4..7]);
        buffer
    }

    /// Regression test: a section declaring more strings than it stored block pointers for must
    /// be rejected at read time instead of panicking on a later ID lookup.
    #[test]
    fn rejects_num_strings_inconsistent_with_block_pointers() {
        let buffer = section_with_patched_preamble(0, 100);
        let mut pos = 0;
        let error =
            PlainFrontCoding::read(&buffer, &mut pos, "test").expect_err("expected an error");
        assert!(matches!(error, HdtError::Malformed(_)), "got {error:?}");
    }

    /// Regression test: a zero block size would make every lookup divide by zero.
    #[test]
    fn rejects_unusable_block_size() {
        let buffer = section_with_patched_preamble(2, 0);
        let mut pos = 0;
        let error =
            PlainFrontCoding::read(&buffer, &mut pos, "test").expect_err("expected an error");
        assert!(matches!(error, HdtError::Malformed(_)), "got {error:?}");
    }

    /// Regression test: even if a section somehow ends up internally inconsistent, resolving an
    /// out-of-range ID must return an error rather than panic inside `BitArray`.
    #[test]
    fn inconsistent_section_errors_instead_of_panicking() {
        let mut pfc = PlainFrontCoding::from_sorted_strings(&["a".to_owned()]);
        pfc.num_strings = 100;
        for id in [17u32, 100] {
            let error = pfc.id_to_string(id).expect_err("expected an error");
            assert!(matches!(error, HdtError::Malformed(_)), "got {error:?}");
        }
        // The string lookups must stay panic-free too.
        assert_eq!(pfc.string_to_id("zzz"), None);
        assert_eq!(pfc.iter().count(), 1);
    }

    #[test]
    fn rejects_wrong_section_type() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(&3u32.to_le_bytes());
        let mut pos = 0;
        let error =
            PlainFrontCoding::read(&buffer, &mut pos, "test").expect_err("expected an error");
        assert!(matches!(error, HdtError::UnsupportedFormat { .. }));
    }
}
