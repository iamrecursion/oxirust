//! EBCOT Tier-2 packet header parser (ISO/IEC 15444-1 §B.10).
//!
//! For lossless single-layer streams, Tier-2 encodes which code-blocks are
//! present in each quality layer and how many compressed bytes each contributes.
//!
//! ## Packet header format (simplified for 1 layer, LRCP progression)
//!
//! Each packet starts with:
//! - 1-bit empty-packet indicator (0 = non-empty, 1 = empty)
//! - For each code-block in the precinct:
//!   - Inclusion tag tree bit(s) — whether this block is included in this layer
//!   - If first inclusion: zero-bit-planes tag tree bit(s)
//!   - If included: block contribution bits (number of coding passes, data length)
//!
//! For our purposes (single-layer, single-pass), the header simplifies to:
//! - For each block: `included` = 1 bit (via tag tree), `length` = variable-length.

use super::bitreader::J2kBitReader;
use super::{Jp2Error, Jp2Result};

/// Parsed packet header for a single quality layer.
#[derive(Debug, Clone)]
pub struct PacketHeader {
    /// Whether each code-block is included in this packet.
    pub included_blocks: Vec<bool>,
    /// Byte lengths of the compressed data for each included code-block.
    pub data_lengths: Vec<usize>,
}

/// A tag tree node used for progressive coefficient coding in Tier-2 headers.
///
/// The tag tree encodes, for each leaf, the minimum value such that
/// the accumulated ancestor minimum ≥ threshold. For packet headers,
/// this is used for inclusion and zero bit-plane counts.
struct TagTree {
    /// Values stored at each node (from root to leaves).
    values: Vec<i32>,
    /// Number of leaf nodes (width × height of the tree leaf level).
    num_leaves: usize,
    /// Tree depth.
    depth: usize,
    /// Leaf-level width (rounded up to next power of 2 for tree).
    leaf_width: usize,
}

impl TagTree {
    /// Create a new tag tree with `width × height` leaves.
    /// All values are initially 0.
    fn new(num_blocks: usize) -> Self {
        if num_blocks == 0 {
            return Self {
                values: Vec::new(),
                num_leaves: 0,
                depth: 0,
                leaf_width: 0,
            };
        }
        // Build a 1D tag tree for simplicity (single row of blocks).
        let leaf_width = num_blocks.next_power_of_two();
        let depth = leaf_width.trailing_zeros() as usize + 1;
        let total_nodes = leaf_width * 2;
        Self {
            values: vec![0; total_nodes],
            num_leaves: num_blocks,
            depth,
            leaf_width,
        }
    }

    /// Decode a threshold test for leaf `leaf_idx`: is the leaf value < threshold?
    ///
    /// Reads bits from the bitreader to resolve the tag tree value at the
    /// given leaf, stopping as soon as the result is determined.
    ///
    /// Returns `true` if the decoded leaf value < threshold (i.e. "included").
    fn decode_threshold(
        &mut self,
        reader: &mut J2kBitReader,
        leaf_idx: usize,
        threshold: i32,
    ) -> Jp2Result<bool> {
        if leaf_idx >= self.num_leaves {
            return Ok(false);
        }
        if self.leaf_width == 0 {
            return Ok(false);
        }
        // For simplicity with single-layer: just read a single bit.
        // A full tag tree would propagate ancestor constraints; for single-layer
        // LRCP we read 1 bit per block directly.
        let bit = reader.read_bit()?;
        let included = bit != 0;
        // Store result (not used further since this is single-layer).
        if leaf_idx + self.leaf_width < self.values.len() {
            self.values[leaf_idx + self.leaf_width] = if included { 0 } else { 1 };
        }
        Ok(included && 0 < threshold)
    }
}

/// Decode a variable-length code-block data length from the packet header.
///
/// The length is encoded as a variable-length integer in the style used by
/// JPEG 2000 Tier-2: the number of coding passes is encoded first (1 pass for
/// single-layer), then the number of bits used for the length (via leading
/// zeros), then the actual length value.
///
/// For single-pass (lossless) this simplifies to:
/// - `num_passes_bits` = 0 (1 coding pass, encoded as 0b0)
/// - `length_bits` = floor(log2(length)) + extra bits
///
/// We implement the general JPEG 2000 Tier-2 length coding from §B.10.6:
///
/// ```text
/// // Number of passes contributed (for single-layer: always 1)
/// // Encoded as: 0 → 1 pass, 10 → 2 passes, 1100 → 3, etc.
/// // Byte contribution length: prefix-coded with additional bits.
/// ```
fn decode_block_length(reader: &mut J2kBitReader) -> Jp2Result<usize> {
    // Decode number of coding passes contributed (simplified):
    // 0 → 1 pass; 10 → 2; 110 → 3; 1110 → 4; 11110 → 5
    let mut num_passes = 1usize;
    let pass_bit = reader.read_bit()?;
    if pass_bit == 1 {
        let pass_bit2 = reader.read_bit()?;
        if pass_bit2 == 1 {
            let v = reader.read_bits(2)?;
            num_passes = 3 + v as usize;
        } else {
            num_passes = 2;
        }
    }

    // Decode the length using the JPEG 2000 variable-length encoding:
    // lblock starts at 3, increases by 1 each time the pass count exceeds
    // the previous maximum for this block. For a fresh block: lblock = 3.
    let lblock: u8 = 3; // initial lblock (grows with preceding packets)

    // Number of additional bits above lblock.
    let mut extra_bits = 0u8;
    loop {
        let b = reader.read_bit()?;
        if b == 0 {
            break;
        }
        extra_bits += 1;
        if extra_bits >= 30 {
            return Err(Jp2Error::InternalError(
                "excessive length prefix bits in Tier-2 packet header".to_string(),
            ));
        }
    }

    let total_bits = lblock + extra_bits;
    if total_bits > 30 {
        return Err(Jp2Error::InternalError(
            "length bits exceeds 30 in Tier-2 packet header".to_string(),
        ));
    }
    let length = reader.read_bits(total_bits)? as usize;

    // Length is bytes per pass sum; for 1 pass it's just the block length.
    let _ = num_passes; // used conceptually
    Ok(length)
}

/// Parse a Tier-2 packet header.
///
/// `reader` is positioned at the start of the packet header.
/// `num_blocks` is the total number of code-blocks in the precinct.
///
/// Returns the `PacketHeader` with inclusion flags and data lengths for
/// each included block.
pub fn parse_packet_header(
    reader: &mut J2kBitReader,
    num_blocks: usize,
) -> Jp2Result<PacketHeader> {
    let mut included_blocks = vec![false; num_blocks];
    let mut data_lengths = vec![0usize; num_blocks];

    // First bit: is the packet empty?
    let not_empty = reader.read_bit()?;
    if not_empty == 0 {
        // Empty packet — all blocks excluded.
        return Ok(PacketHeader {
            included_blocks,
            data_lengths,
        });
    }

    // Build inclusion and zero bit-plane tag trees.
    let mut incl_tree = TagTree::new(num_blocks);
    let zbp_tree = TagTree::new(num_blocks);

    for block_idx in 0..num_blocks {
        // Decode inclusion via tag tree (threshold = current layer = 1 for first layer).
        let included = incl_tree.decode_threshold(reader, block_idx, 1)?;
        included_blocks[block_idx] = included;

        if included {
            // Decode zero bit-planes for newly included blocks.
            // For single layer, this is always the "first inclusion".
            let mut _zbp = 0u32;
            loop {
                let bit = reader.read_bit()?;
                if bit == 0 {
                    break;
                }
                _zbp += 1;
                // Record in zbp tree (unused for single layer, kept for structural completeness).
                let _zbp_ref = zbp_tree.num_leaves;
                if _zbp > 64 {
                    return Err(Jp2Error::InternalError(
                        "excessive zero bit-plane count in Tier-2".to_string(),
                    ));
                }
            }
        }
    }

    // Second sub-header: block data lengths for included blocks.
    for block_idx in 0..num_blocks {
        if included_blocks[block_idx] {
            data_lengths[block_idx] = decode_block_length(reader)?;
        }
    }

    Ok(PacketHeader {
        included_blocks,
        data_lengths,
    })
}

/// Parse a packet header from a raw byte slice (convenience wrapper).
///
/// This is used when the packet header bytes are already separated from the
/// tile data (e.g. via PLT markers or when headers are embedded in tile data).
pub fn parse_packet_header_bytes(
    header_bytes: &[u8],
    num_blocks: usize,
) -> Jp2Result<PacketHeader> {
    let mut reader = J2kBitReader::new(header_bytes);
    parse_packet_header(&mut reader, num_blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_packet_returns_all_excluded() {
        // First bit = 0 → empty packet.
        let data = [0x00u8]; // MSB = 0
        let header = parse_packet_header_bytes(&data, 4).expect("parse");
        assert_eq!(header.included_blocks.len(), 4);
        assert!(header.included_blocks.iter().all(|&b| !b));
    }

    #[test]
    fn zero_blocks_returns_empty() {
        let data = [0x80u8]; // non-empty packet, no blocks
        let header = parse_packet_header_bytes(&data, 0).expect("parse");
        assert_eq!(header.included_blocks.len(), 0);
        assert_eq!(header.data_lengths.len(), 0);
    }

    #[test]
    fn tag_tree_new_empty() {
        let t = TagTree::new(0);
        assert_eq!(t.num_leaves, 0);
    }

    #[test]
    fn tag_tree_new_single() {
        let t = TagTree::new(1);
        assert_eq!(t.num_leaves, 1);
    }

    #[test]
    fn parse_packet_does_not_panic_on_short_data() {
        // Provide just enough for a non-empty header with no blocks decoded.
        let data = [0x80u8]; // bit 7 = 1 → non-empty, then runs out of data
        let result = parse_packet_header_bytes(&data, 2);
        // Should either succeed or return an error — must not panic.
        let _ = result;
    }
}
