//! Vorbis codebook (Huffman VQ book) abstraction.
//!
//! Vorbis uses custom Huffman books built from entries with associated vector
//! dimensions. This module implements the essential codebook structure and
//! lookup operations.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

/// A single codebook entry.
#[derive(Clone, Debug)]
pub struct CodebookEntry {
    /// Huffman code length in bits (0 = unused entry).
    pub length: u8,
    /// The scalar or vector value(s) decoded when this entry is hit.
    pub value: Vec<f32>,
}

/// A Vorbis codebook.
#[derive(Clone, Debug)]
pub struct Codebook {
    /// Number of entries.
    pub entries: usize,
    /// Entry descriptors.
    pub table: Vec<CodebookEntry>,
    /// Vector dimension (1 for scalar books).
    pub dimensions: usize,
}

impl Codebook {
    /// Build a codebook from entry lengths and associated scalar values.
    ///
    /// `lengths[i]` is the Huffman code length for entry `i`.
    /// `values[i]` is the decoded scalar value for entry `i` (dimension=1).
    #[must_use]
    pub fn from_lengths_and_values(lengths: &[u8], values: &[f32]) -> Self {
        let entries = lengths.len();
        let table = (0..entries)
            .map(|i| CodebookEntry {
                length: lengths[i],
                value: vec![values[i.min(values.len() - 1)]],
            })
            .collect();
        Self {
            entries,
            table,
            dimensions: 1,
        }
    }

    /// Look up the value for the entry with the given index.
    ///
    /// Returns `None` if the index is out of range or the entry is unused (length=0).
    #[must_use]
    pub fn lookup(&self, index: usize) -> Option<&[f32]> {
        self.table
            .get(index)
            .filter(|e| e.length > 0)
            .map(|e| e.value.as_slice())
    }

    /// Compute the average code length (entropy estimate).
    #[must_use]
    pub fn average_code_length(&self) -> f64 {
        let active: Vec<u8> = self
            .table
            .iter()
            .filter(|e| e.length > 0)
            .map(|e| e.length)
            .collect();
        if active.is_empty() {
            return 0.0;
        }
        active.iter().map(|&l| f64::from(l)).sum::<f64>() / active.len() as f64
    }

    /// Kraft inequality check: returns `true` if code lengths satisfy Kraft's inequality.
    #[must_use]
    pub fn kraft_inequality_satisfied(&self) -> bool {
        let sum: f64 = self
            .table
            .iter()
            .filter(|e| e.length > 0)
            .map(|e| 2.0f64.powi(-(e.length as i32)))
            .sum();
        sum <= 1.0 + 1e-9
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_book() -> Codebook {
        let lengths = [1u8, 2, 3, 3];
        let values = [0.0f32, 1.0, 2.0, 3.0];
        Codebook::from_lengths_and_values(&lengths, &values)
    }

    #[test]
    fn test_codebook_entry_count() {
        let book = simple_book();
        assert_eq!(book.entries, 4);
    }

    #[test]
    fn test_codebook_lookup_valid() {
        let book = simple_book();
        let v = book.lookup(0).expect("entry 0 should exist");
        assert!((v[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_codebook_lookup_out_of_range() {
        let book = simple_book();
        assert!(book.lookup(10).is_none());
    }

    #[test]
    fn test_codebook_average_length() {
        let book = simple_book();
        let avg = book.average_code_length();
        // Lengths: 1, 2, 3, 3 → avg = (1+2+3+3)/4 = 2.25
        assert!((avg - 2.25).abs() < 1e-9, "Expected 2.25, got {avg}");
    }

    #[test]
    fn test_codebook_kraft_inequality() {
        let book = simple_book();
        // 2^-1 + 2^-2 + 2^-3 + 2^-3 = 0.5+0.25+0.125+0.125 = 1.0
        assert!(book.kraft_inequality_satisfied());
    }

    #[test]
    fn test_codebook_unused_entry_length_zero() {
        let lengths = [0u8, 2, 0, 3];
        let values = [0.0f32; 4];
        let book = Codebook::from_lengths_and_values(&lengths, &values);
        assert!(book.lookup(0).is_none()); // length=0 → unused
        assert!(book.lookup(1).is_some());
        assert!(book.lookup(2).is_none());
    }

    #[test]
    fn test_codebook_dimension_is_one() {
        let book = simple_book();
        assert_eq!(book.dimensions, 1);
    }
}
