//! Vorbis codebook structures.
//!
//! Codebooks are used for entropy coding in Vorbis. They consist of:
//! - A Huffman tree for codeword decoding
//! - An optional vector quantization (VQ) lookup table
//!
//! # Codebook Types
//!
//! - Type 0: Scalar codebook (no VQ)
//! - Type 1: Lattice VQ (implicit values, Vorbis spec §3.2.1)
//! - Type 2: Tessellated VQ (explicit values)

#![forbid(unsafe_code)]

use super::bitpack::BitReader;
use crate::AudioError;

/// Codebook entry containing codeword and optional VQ value.
#[derive(Debug, Clone, Default)]
pub struct CodebookEntry {
    /// Codeword length in bits.
    pub length: u8,
    /// Codeword value.
    pub codeword: u32,
    /// Entry used flag.
    pub used: bool,
    /// VQ lookup values (if applicable).
    pub values: Vec<f32>,
}

impl CodebookEntry {
    /// Create a new unused entry.
    #[must_use]
    pub fn unused() -> Self {
        Self {
            length: 0,
            codeword: 0,
            used: false,
            values: Vec::new(),
        }
    }

    /// Create a new used entry with given length.
    #[must_use]
    pub fn new(length: u8) -> Self {
        Self {
            length,
            codeword: 0,
            used: length > 0,
            values: Vec::new(),
        }
    }

    /// Check if this entry is valid for decoding.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.used && self.length > 0
    }
}

// ─────────────────────────────────────────────────────────────────
// HuffmanNode / HuffmanTree
// ─────────────────────────────────────────────────────────────────

/// Huffman tree node.
#[derive(Debug, Clone)]
pub enum HuffmanNode {
    /// Internal node with left (bit=0) and right (bit=1) children.
    Internal {
        /// Left child (bit 0).
        left: Box<HuffmanNode>,
        /// Right child (bit 1).
        right: Box<HuffmanNode>,
    },
    /// Leaf node carrying an entry index.
    Leaf(usize),
    /// Empty placeholder (unoccupied subtree).
    Empty,
}

impl Default for HuffmanNode {
    fn default() -> Self {
        Self::Empty
    }
}

/// Huffman tree for codeword decoding.
///
/// Built from per-entry code lengths using the canonical Huffman
/// assignment algorithm described in Vorbis spec §3.2.
#[derive(Debug, Clone, Default)]
pub struct HuffmanTree {
    /// Root node.
    root: HuffmanNode,
    /// Maximum code length.
    max_length: u8,
    /// Number of entries.
    entry_count: usize,
}

impl HuffmanTree {
    /// Create a new empty Huffman tree.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: HuffmanNode::Empty,
            max_length: 0,
            entry_count: 0,
        }
    }

    /// Build a canonical Huffman tree from per-entry code lengths.
    ///
    /// Entries with `length == 0` are treated as unused (sparse codebooks).
    /// The canonical assignment follows Vorbis spec §3.2: sort by (length, entry
    /// index), then assign codes in ascending order.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidData`] if the lengths violate the Kraft
    /// inequality (i.e., the codebook is over-full).
    pub fn build(lengths: &[u8]) -> Result<Self, AudioError> {
        if lengths.is_empty() {
            return Ok(Self::new());
        }

        let max_length = *lengths.iter().max().unwrap_or(&0);
        if max_length > 32 {
            return Err(AudioError::InvalidData("Code length too long".into()));
        }
        if max_length == 0 {
            // All lengths are zero — empty codebook.
            return Ok(Self {
                root: HuffmanNode::Empty,
                max_length: 0,
                entry_count: lengths.len(),
            });
        }

        // Validate using Kraft inequality (sum of 2^(max_len - len_i) ≤ 2^max_len).
        let kraft_sum: u64 = lengths
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1u64 << (max_length - l))
            .sum();

        if kraft_sum > (1u64 << max_length) {
            return Err(AudioError::InvalidData(
                "Invalid Huffman code lengths (Kraft inequality violated)".into(),
            ));
        }

        let entry_count = lengths.len();

        // Collect (length, entry_index) pairs for used entries and sort
        // canonically: primary key = length, secondary key = index.
        let mut entries: Vec<(u8, usize)> = lengths
            .iter()
            .enumerate()
            .filter(|(_, &l)| l > 0)
            .map(|(i, &l)| (l, i))
            .collect();
        entries.sort_unstable();

        // Assign canonical codewords and insert into the binary tree.
        let mut root = HuffmanNode::Empty;
        let mut current_code: u32 = 0;
        let mut current_length: u8 = 0;

        for (len, entry_idx) in &entries {
            let len = *len;
            let entry_idx = *entry_idx;

            // Shift code to the new length (extend with zeros).
            if len > current_length {
                current_code <<= len - current_length;
                current_length = len;
            }

            // Insert leaf at the current codeword.
            Self::insert(&mut root, current_code, len, entry_idx)?;

            // Advance to next code (canonical increment).
            current_code += 1;
        }

        Ok(Self {
            root,
            max_length,
            entry_count,
        })
    }

    /// Insert a leaf at the path described by `code` (MSB-first, `len` bits).
    fn insert(
        node: &mut HuffmanNode,
        code: u32,
        len: u8,
        entry_idx: usize,
    ) -> Result<(), AudioError> {
        if len == 0 {
            // We have arrived at the destination.
            match node {
                HuffmanNode::Empty => {
                    *node = HuffmanNode::Leaf(entry_idx);
                    Ok(())
                }
                HuffmanNode::Leaf(_) => Err(AudioError::InvalidData(
                    "Huffman tree: duplicate codeword".into(),
                )),
                HuffmanNode::Internal { .. } => Err(AudioError::InvalidData(
                    "Huffman tree: codeword prefix conflict".into(),
                )),
            }
        } else {
            // Navigate left (bit=0) or right (bit=1) based on MSB.
            let bit = (code >> (len - 1)) & 1;
            let rest_code = code & ((1u32 << (len - 1)) - 1);
            let rest_len = len - 1;

            // Ensure this node is Internal.
            if matches!(node, HuffmanNode::Leaf(_)) {
                return Err(AudioError::InvalidData(
                    "Huffman tree: codeword prefix is already a leaf".into(),
                ));
            }
            if matches!(node, HuffmanNode::Empty) {
                *node = HuffmanNode::Internal {
                    left: Box::new(HuffmanNode::Empty),
                    right: Box::new(HuffmanNode::Empty),
                };
            }

            match node {
                HuffmanNode::Internal { left, right } => {
                    if bit == 0 {
                        Self::insert(left, rest_code, rest_len, entry_idx)
                    } else {
                        Self::insert(right, rest_code, rest_len, entry_idx)
                    }
                }
                _ => unreachable!("node was just set to Internal"),
            }
        }
    }

    /// Get maximum code length.
    #[must_use]
    pub fn max_length(&self) -> u8 {
        self.max_length
    }

    /// Get number of entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Decode one symbol from `bits` by traversing the Huffman tree.
    ///
    /// Reads one bit at a time (LSB-first, matching Vorbis spec):
    /// - bit `0` → go left
    /// - bit `1` → go right
    ///
    /// Returns the entry index stored in the matching leaf.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Eof`] when the bitstream runs out, or
    /// [`AudioError::InvalidData`] when no leaf is reachable.
    pub fn decode(&self, bits: &mut BitReader<'_>) -> Result<usize, AudioError> {
        let mut node = &self.root;

        loop {
            match node {
                HuffmanNode::Leaf(idx) => return Ok(*idx),
                HuffmanNode::Empty => {
                    return Err(AudioError::InvalidData(
                        "Huffman decode: reached empty node (invalid bitstream)".into(),
                    ))
                }
                HuffmanNode::Internal { left, right } => {
                    let bit = bits.read_bit()?;
                    node = if bit { right } else { left };
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// LookupType
// ─────────────────────────────────────────────────────────────────

/// Codebook lookup type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LookupType {
    /// No lookup (scalar codebook).
    #[default]
    None,
    /// Type 1: Lattice VQ.
    Lattice,
    /// Type 2: Tessellated VQ.
    Tessellated,
}

impl LookupType {
    /// Create from raw value.
    #[must_use]
    pub fn from_value(value: u8) -> Option<Self> {
        match value {
            0 => Some(LookupType::None),
            1 => Some(LookupType::Lattice),
            2 => Some(LookupType::Tessellated),
            _ => None,
        }
    }

    /// Check if this type has VQ values.
    #[must_use]
    pub fn has_lookup(self) -> bool {
        self != LookupType::None
    }
}

// ─────────────────────────────────────────────────────────────────
// Codebook
// ─────────────────────────────────────────────────────────────────

/// Vorbis codebook.
#[derive(Debug, Clone, Default)]
pub struct Codebook {
    /// Codebook identifier/index.
    pub id: usize,
    /// Number of entries.
    pub entries: usize,
    /// Entry dimensions (for VQ).
    pub dimensions: u16,
    /// Huffman tree for decoding.
    pub tree: HuffmanTree,
    /// Lookup type.
    pub lookup_type: LookupType,
    /// Minimum value for VQ.
    pub minimum_value: f32,
    /// Delta value for VQ.
    pub delta_value: f32,
    /// Value bits for VQ.
    pub value_bits: u8,
    /// Sequence flag for VQ.
    pub sequence_p: bool,
    /// Multiplicands for VQ lookup.
    pub multiplicands: Vec<u32>,
    /// Codebook entries.
    pub entry_list: Vec<CodebookEntry>,
}

impl Codebook {
    /// Create a new empty codebook.
    #[must_use]
    pub fn new(id: usize) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }

    /// Parse codebook from byte slice (skeleton — bit-level parsing not yet implemented).
    ///
    /// # Errors
    ///
    /// Returns error if codebook data is too short.
    pub fn parse(id: usize, data: &[u8]) -> Result<Self, AudioError> {
        // Check sync pattern (0x564342 = "BCV").
        if data.len() < 10 {
            return Err(AudioError::InvalidData("Codebook data too short".into()));
        }

        // Skeleton: actual parsing requires bit-level reading.
        Ok(Self::new(id))
    }

    /// Get number of entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries
    }

    /// Get dimensions.
    #[must_use]
    pub fn dimensions(&self) -> u16 {
        self.dimensions
    }

    /// Decode a single scalar value from the bitstream (Vorbis spec §3.2).
    ///
    /// Uses the codebook's Huffman tree to read one codeword, maps the resulting
    /// entry index to a floating-point scalar via:
    ///
    /// ```text
    /// value = minimum_value + multiplicand[index] * delta_value
    /// ```
    ///
    /// When the multiplicands table is empty (pure-scalar codebook with no VQ
    /// table), the entry index itself is used as the multiplicand, which is the
    /// correct behaviour for codebooks whose lookup_type is `None`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidData`] if the entry index is out of range or
    /// the bitstream is invalid.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn decode_scalar(&self, bits: &mut BitReader<'_>) -> Result<f32, AudioError> {
        let index = self.tree.decode(bits)?;

        if self.entries > 0 && index >= self.entries {
            return Err(AudioError::InvalidData(format!(
                "scalar entry index out of range: {index} >= {}",
                self.entries
            )));
        }

        // Per Vorbis spec §3.2: value = minimum_value + multiplicand * delta_value.
        // When the multiplicands table is populated, look up by index.
        // For lookup_type == None (scalar-only), multiplicands is empty and we
        // use the raw index as the multiplicand (encodes the ordinal directly).
        let multiplicand = self
            .multiplicands
            .get(index)
            .copied()
            .unwrap_or(index as u32);

        Ok(self.minimum_value + (multiplicand as f32) * self.delta_value)
    }

    /// Decode a VQ vector from the bitstream (Vorbis spec §3.2 / §3.2.1).
    ///
    /// Reads a Huffman-coded entry index, then reconstructs a floating-point
    /// vector of length [`Self::dimensions`] using the codebook's lookup table.
    ///
    /// This delegates to [`Self::lookup`] once the entry index is known, so both
    /// Lattice (type 1) and Tessellated (type 2) codebooks are handled correctly.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidData`] if:
    /// - The codebook has no VQ lookup (`lookup_type == None`).
    /// - The decoded entry index is out of range.
    /// - The bitstream is invalid.
    pub fn decode_vq(&self, bits: &mut BitReader<'_>) -> Result<Vec<f32>, AudioError> {
        if !self.lookup_type.has_lookup() {
            return Err(AudioError::InvalidData("Codebook has no VQ lookup".into()));
        }

        let index = self.tree.decode(bits)?;

        self.lookup(index).ok_or_else(|| {
            AudioError::InvalidData(format!(
                "VQ lookup out of range: index {index}, entries {}",
                self.entries
            ))
        })
    }

    /// Look up the VQ vector for the given entry index.
    ///
    /// Returns `None` when the codebook has no lookup or `index >= entries`.
    ///
    /// # Type 1 (Lattice VQ)
    ///
    /// Values are derived from the multiplicands table using a mixed-radix
    /// base decomposition (Vorbis spec §3.2.1):
    ///
    /// ```text
    /// lookup_offset = index
    /// for d in 0..dimensions:
    ///     values[d] = minimum_value + multiplicands[lookup_offset % len] * delta_value
    ///     lookup_offset /= len
    /// ```
    ///
    /// # Type 2 (Tessellated VQ)
    ///
    /// Values are stored explicitly:
    ///
    /// ```text
    /// values[d] = minimum_value + multiplicands[index * dimensions + d] * delta_value
    /// ```
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn lookup(&self, index: usize) -> Option<Vec<f32>> {
        if !self.lookup_type.has_lookup() || index >= self.entries {
            return None;
        }

        match self.lookup_type {
            LookupType::Lattice => {
                // Type 1: compute values from multiplicands using mixed-radix decomposition.
                let lookup_values = self.multiplicands.len();
                if lookup_values == 0 {
                    return None;
                }
                let mut values = Vec::with_capacity(self.dimensions as usize);
                let mut lookup_offset = index;

                for _ in 0..self.dimensions {
                    let multiplicand = self.multiplicands[lookup_offset % lookup_values];
                    let value = self.minimum_value + (multiplicand as f32) * self.delta_value;
                    values.push(value);
                    lookup_offset /= lookup_values;
                }
                Some(values)
            }
            LookupType::Tessellated => {
                // Type 2: values stored directly, dimensions entries per index.
                let start = index * self.dimensions as usize;
                let end = start + self.dimensions as usize;
                if end > self.multiplicands.len() {
                    return None;
                }
                let values: Vec<f32> = self.multiplicands[start..end]
                    .iter()
                    .map(|&m| self.minimum_value + (m as f32) * self.delta_value)
                    .collect();
                Some(values)
            }
            LookupType::None => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// CodebookSet
// ─────────────────────────────────────────────────────────────────

/// Codebook collection for a Vorbis stream.
#[derive(Debug, Clone, Default)]
pub struct CodebookSet {
    /// All codebooks in the stream.
    codebooks: Vec<Codebook>,
}

impl CodebookSet {
    /// Create a new empty codebook set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            codebooks: Vec::new(),
        }
    }

    /// Add a codebook to the set.
    pub fn add(&mut self, codebook: Codebook) {
        self.codebooks.push(codebook);
    }

    /// Get codebook by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Codebook> {
        self.codebooks.get(index)
    }

    /// Get number of codebooks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.codebooks.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codebooks.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::bitpack::{BitPacker, BitReader};
    use super::*;

    // ── CodebookEntry ────────────────────────────────────────────

    #[test]
    fn test_codebook_entry_unused() {
        let entry = CodebookEntry::unused();
        assert!(!entry.used);
        assert!(!entry.is_valid());
    }

    #[test]
    fn test_codebook_entry_new() {
        let entry = CodebookEntry::new(5);
        assert!(entry.used);
        assert_eq!(entry.length, 5);
        assert!(entry.is_valid());
    }

    #[test]
    fn test_codebook_entry_zero_length() {
        let entry = CodebookEntry::new(0);
        assert!(!entry.used);
        assert!(!entry.is_valid());
    }

    // ── HuffmanTree metadata ─────────────────────────────────────

    #[test]
    fn test_huffman_tree_new() {
        let tree = HuffmanTree::new();
        assert_eq!(tree.max_length(), 0);
        assert_eq!(tree.entry_count(), 0);
    }

    #[test]
    fn test_huffman_tree_build() {
        let lengths = vec![2, 2, 3, 3, 3, 3];
        let tree = HuffmanTree::build(&lengths).expect("should succeed");
        assert_eq!(tree.max_length(), 3);
        assert_eq!(tree.entry_count(), 6);
    }

    #[test]
    fn test_huffman_tree_empty() {
        let tree = HuffmanTree::build(&[]).expect("should succeed");
        assert_eq!(tree.max_length(), 0);
        assert_eq!(tree.entry_count(), 0);
    }

    // ── HuffmanTree::decode ───────────────────────────────────────

    /// Build a small canonical Huffman tree and verify round-trip decode.
    ///
    /// Lengths: [1, 2, 3, 3] — canonical codes:
    ///   entry 0: 0       (1 bit,  MSB-first code = 0)
    ///   entry 1: 10      (2 bits, MSB-first = 10)
    ///   entry 2: 110     (3 bits, MSB-first = 110)
    ///   entry 3: 111     (3 bits, MSB-first = 111)
    ///
    /// Packed LSB-first in a BitPacker for decoding.
    #[test]
    fn test_huffman_decode_basic() {
        let lengths = vec![1u8, 2, 3, 3];
        let tree = HuffmanTree::build(&lengths).expect("build ok");

        // Encode entry 0 (code "0", 1 bit) then entry 1 (code "10", 2 bits)
        // then entry 2 (code "110", 3 bits).
        // LSB-first packing:
        //   entry 0: write bit 0 (MSB of "0" is bit 0 → write 0)
        //   entry 1: write bit 0 then bit 1  (MSB first "1","0" → LSB-first: 0,1)
        //   entry 2: write 0,1,1 (MSB-first "1","1","0" → LSB-first: 0,1,1)
        let mut packer = BitPacker::new();
        // entry 0: code=0b0, len=1  → LSB-first: bit0=0
        packer.write_bits(0b0, 1);
        // entry 1: code=0b10, len=2 → LSB-first: bit0=0, bit1=1
        packer.write_bits(0b01, 2);
        // entry 2: code=0b110, len=3 → LSB-first: bit0=0, bit1=1, bit2=1
        packer.write_bits(0b011, 3);
        let bytes = packer.finish();

        let mut reader = BitReader::new(&bytes);
        assert_eq!(tree.decode(&mut reader).expect("entry 0"), 0);
        assert_eq!(tree.decode(&mut reader).expect("entry 1"), 1);
        assert_eq!(tree.decode(&mut reader).expect("entry 2"), 2);
    }

    /// Decoding an invalid bit sequence should return an error.
    #[test]
    fn test_huffman_decode_eof_error() {
        // Tree: entry 0 has a 3-bit codeword.
        let lengths = vec![3u8];
        let tree = HuffmanTree::build(&lengths).expect("build ok");

        // Only 2 bits available — not enough to reach any leaf.
        let data = [0x00u8];
        let mut reader = BitReader::new(&data);
        let _ = reader.read_bits(6); // exhaust 6 bits
                                     // Only 2 bits left; trying to read bit 3 should fail.
        assert!(tree.decode(&mut reader).is_err());
    }

    /// Single-entry codebook (length 1): always decodes the same symbol.
    #[test]
    fn test_huffman_decode_single_entry() {
        let lengths = vec![1u8];
        let tree = HuffmanTree::build(&lengths).expect("build ok");

        let mut packer = BitPacker::new();
        packer.write_bits(0b0, 1); // code for entry 0 is "0"
        let bytes = packer.finish();

        let mut reader = BitReader::new(&bytes);
        assert_eq!(tree.decode(&mut reader).expect("entry 0"), 0);
    }

    // ── LookupType ───────────────────────────────────────────────

    #[test]
    fn test_lookup_type() {
        assert_eq!(LookupType::from_value(0), Some(LookupType::None));
        assert_eq!(LookupType::from_value(1), Some(LookupType::Lattice));
        assert_eq!(LookupType::from_value(2), Some(LookupType::Tessellated));
        assert_eq!(LookupType::from_value(3), None);
    }

    #[test]
    fn test_lookup_type_has_lookup() {
        assert!(!LookupType::None.has_lookup());
        assert!(LookupType::Lattice.has_lookup());
        assert!(LookupType::Tessellated.has_lookup());
    }

    // ── Codebook metadata ────────────────────────────────────────

    #[test]
    fn test_codebook_new() {
        let codebook = Codebook::new(0);
        assert_eq!(codebook.id, 0);
        assert_eq!(codebook.entry_count(), 0);
    }

    #[test]
    fn test_codebook_set() {
        let mut set = CodebookSet::new();
        assert!(set.is_empty());

        set.add(Codebook::new(0));
        set.add(Codebook::new(1));

        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
        assert!(set.get(0).is_some());
        assert!(set.get(1).is_some());
        assert!(set.get(2).is_none());
    }

    #[test]
    fn test_codebook_lookup_no_vq() {
        let codebook = Codebook::new(0);
        assert!(codebook.lookup(0).is_none());
    }

    // ── Codebook::decode_scalar ───────────────────────────────────

    /// Verify decode_scalar with a populated multiplicands table (type-1 path).
    ///
    /// decode_scalar returns `minimum_value + multiplicand * delta_value` (f32).
    /// Entry 0 → multiplicands[0] = 0 → 10.0 + 0 × 5.0 = 10.0
    /// Entry 1 → multiplicands[1] = 1 → 10.0 + 1 × 5.0 = 15.0
    #[test]
    fn test_decode_scalar_with_multiplicands() {
        // 2-entry codebook, lengths [1, 1].
        // Canonical codes: entry 0 → "0" (1 bit), entry 1 → "1" (1 bit).
        let lengths = vec![1u8, 1];
        let tree = HuffmanTree::build(&lengths).expect("build ok");

        let mut cb = Codebook::new(0);
        cb.entries = 2;
        cb.tree = tree;
        cb.lookup_type = LookupType::None;
        cb.minimum_value = 10.0;
        cb.delta_value = 5.0;
        cb.multiplicands = vec![0, 1];

        // Encode: bit 0 → entry 0; bit 1 → entry 1.
        let mut packer = BitPacker::new();
        packer.write_bits(0b0, 1);
        packer.write_bits(0b1, 1);
        let bytes = packer.finish();

        let mut reader = BitReader::new(&bytes);
        let v0 = cb.decode_scalar(&mut reader).expect("entry 0");
        assert!(
            (v0 - 10.0_f32).abs() < 1e-6,
            "entry 0 expected 10.0, got {v0}"
        );
        let v1 = cb.decode_scalar(&mut reader).expect("entry 1");
        assert!(
            (v1 - 15.0_f32).abs() < 1e-6,
            "entry 1 expected 15.0, got {v1}"
        );
    }

    /// Verify decode_scalar when multiplicands table is empty (type-0 / scalar path).
    ///
    /// When no multiplicands table is present, the raw entry index is used as
    /// the multiplicand: value = minimum_value + index × delta_value.
    /// Entry 0 → 2.0 + 0 × 3.0 = 2.0
    /// Entry 1 → 2.0 + 1 × 3.0 = 5.0
    #[test]
    fn test_decode_scalar_no_multiplicands_uses_index() {
        let lengths = vec![1u8, 1];
        let tree = HuffmanTree::build(&lengths).expect("build ok");

        let mut cb = Codebook::new(0);
        cb.entries = 2;
        cb.tree = tree;
        cb.lookup_type = LookupType::None;
        cb.minimum_value = 2.0;
        cb.delta_value = 3.0;
        // multiplicands is intentionally empty — the fallback uses index as multiplicand.

        let mut packer = BitPacker::new();
        packer.write_bits(0b0, 1); // entry 0
        packer.write_bits(0b1, 1); // entry 1
        let bytes = packer.finish();

        let mut reader = BitReader::new(&bytes);
        let v0 = cb.decode_scalar(&mut reader).expect("entry 0");
        assert!(
            (v0 - 2.0_f32).abs() < 1e-6,
            "entry 0: expected 2.0, got {v0}"
        );
        let v1 = cb.decode_scalar(&mut reader).expect("entry 1");
        assert!(
            (v1 - 5.0_f32).abs() < 1e-6,
            "entry 1: expected 5.0, got {v1}"
        );
    }

    // ── Codebook::decode_vq ───────────────────────────────────────

    /// Lattice VQ (type 1): verify the reconstructed vector.
    ///
    /// 4-entry lattice codebook, 2 dimensions, 2 multiplicands.
    /// Lengths: [2,2,2,2] — canonical codes 00,01,10,11.
    ///
    /// Multiplicands: [0, 1]
    /// minimum_value = 0.0, delta_value = 1.0
    ///
    /// Mixed-radix decomposition (lookup_values = 2):
    ///   index 0 → [mults[0%2], mults[0/2 % 2]] = [0,0] → [0.0, 0.0]
    ///   index 1 → [mults[1%2], mults[1/2 % 2]] = [1,0] → [1.0, 0.0]
    ///   index 2 → [mults[2%2], mults[2/2 % 2]] = [0,1] → [0.0, 1.0]
    ///   index 3 → [mults[3%2], mults[3/2 % 2]] = [1,1] → [1.0, 1.0]
    #[test]
    fn test_decode_vq_lattice() {
        let lengths = vec![2u8, 2, 2, 2];
        let tree = HuffmanTree::build(&lengths).expect("build ok");

        let mut cb = Codebook::new(0);
        cb.entries = 4;
        cb.dimensions = 2;
        cb.tree = tree;
        cb.lookup_type = LookupType::Lattice;
        cb.minimum_value = 0.0;
        cb.delta_value = 1.0;
        cb.multiplicands = vec![0, 1];

        // Canonical codes for lengths [2,2,2,2]:
        //   entry 0 → 0b00, entry 1 → 0b01, entry 2 → 0b10, entry 3 → 0b11
        // LSB-first: 0b00 → write bits 0,0; 0b01 → write bits 1,0; etc.

        // We decode index 2 (code = 0b10, LSB-first: 0,1).
        let mut packer = BitPacker::new();
        packer.write_bits(0b01, 2); // entry 2 LSB-first: bit0=0, bit1=1
        let bytes = packer.finish();

        let mut reader = BitReader::new(&bytes);
        let vec = cb.decode_vq(&mut reader).expect("decode vq");
        assert_eq!(vec.len(), 2);
        assert!(
            (vec[0] - 0.0).abs() < 1e-6,
            "dim 0 should be 0.0, got {}",
            vec[0]
        );
        assert!(
            (vec[1] - 1.0).abs() < 1e-6,
            "dim 1 should be 1.0, got {}",
            vec[1]
        );
    }

    /// Tessellated VQ (type 2): explicit multiplicand table.
    #[test]
    fn test_decode_vq_tessellated() {
        // 2-entry, 2-dimension codebook, lengths [1,1].
        // multiplicands: [10, 20, 30, 40]
        //   index 0 → multiplicands[0..2] = [10, 20] → [0.0 + 10*0.5, 0.0 + 20*0.5] = [5.0, 10.0]
        //   index 1 → multiplicands[2..4] = [30, 40] → [15.0, 20.0]
        let lengths = vec![1u8, 1];
        let tree = HuffmanTree::build(&lengths).expect("build ok");

        let mut cb = Codebook::new(0);
        cb.entries = 2;
        cb.dimensions = 2;
        cb.tree = tree;
        cb.lookup_type = LookupType::Tessellated;
        cb.minimum_value = 0.0;
        cb.delta_value = 0.5;
        cb.multiplicands = vec![10, 20, 30, 40];

        // Decode index 1 (code = "1", 1 bit, LSB-first bit = 1).
        let mut packer = BitPacker::new();
        packer.write_bits(0b1, 1);
        let bytes = packer.finish();

        let mut reader = BitReader::new(&bytes);
        let vec = cb.decode_vq(&mut reader).expect("decode vq");
        assert_eq!(vec.len(), 2);
        assert!((vec[0] - 15.0).abs() < 1e-6, "dim 0 got {}", vec[0]);
        assert!((vec[1] - 20.0).abs() < 1e-6, "dim 1 got {}", vec[1]);
    }

    /// decode_vq on a scalar codebook (no lookup) must return an error.
    #[test]
    fn test_decode_vq_no_lookup_error() {
        let mut cb = Codebook::new(0);
        cb.lookup_type = LookupType::None;
        let data = [0xFFu8];
        let mut reader = BitReader::new(&data);
        assert!(cb.decode_vq(&mut reader).is_err());
    }

    // ── Kraft inequality enforcement ──────────────────────────────

    #[test]
    fn test_huffman_build_kraft_violation() {
        // All lengths 1 for 3 entries → sum 3 * 2^(1-1) = 3 > 2 — over-full.
        let lengths = vec![1u8, 1, 1];
        assert!(HuffmanTree::build(&lengths).is_err());
    }
}
