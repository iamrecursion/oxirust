//! EBCOT Tier-2 packet *encoder* (ISO/IEC 15444-1 §B.10) — the exact forward
//! counterpart of [`super::tier2`].
//!
//! Produces a single-quality-layer packet for one tile-component: the packet
//! header (empty-packet flag, per-code-block inclusion bit, zero-bit-plane
//! count, and code-block byte length) followed by the byte-aligned concatenation
//! of the code-block MQ streams. The header bit layout mirrors
//! [`super::tier2::parse_packet_header`] field-for-field, including the
//! `lblock = 3` length coding and the JPEG 2000 bit-stuffing rule reversed by
//! [`super::bitreader::J2kBitReader`] (after a `0xFF` byte the next byte's top
//! bit is a stuffed zero).

use super::{Jp2Error, Jp2Result};

/// MSB-first bit writer that mirrors [`super::bitreader::J2kBitReader`],
/// inserting a stuffed zero bit after every `0xFF` byte so that no `0xFF 0xXX`
/// (XX > 0x8F) marker pattern can appear in the header.
pub struct J2kBitWriter {
    /// Output bytes produced so far.
    out: Vec<u8>,
    /// Byte under construction.
    cur: u8,
    /// Number of bits already placed in `cur` (from the MSB downward).
    nbits: u8,
    /// Number of bit positions available in the current byte (7 right after a
    /// `0xFF` byte due to stuffing, otherwise 8).
    capacity: u8,
    /// Whether the most-recently flushed byte was `0xFF`.
    last_was_ff: bool,
}

impl Default for J2kBitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl J2kBitWriter {
    /// Create an empty bit writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            cur: 0,
            nbits: 0,
            capacity: 8,
            last_was_ff: false,
        }
    }

    /// Write a single bit (the low bit of `bit`).
    pub fn write_bit(&mut self, bit: u8) {
        // `cur` accumulates bits from the MSB of the available window downward.
        // With a 7-bit window (post-0xFF stuffing) the top bit stays 0.
        let shift = self.capacity - 1 - self.nbits;
        self.cur |= (bit & 1) << shift;
        self.nbits += 1;
        if self.nbits == self.capacity {
            self.flush_byte();
        }
    }

    /// Write the low `n` bits of `value`, most-significant first.
    pub fn write_bits(&mut self, value: u32, n: u8) {
        let mut i = n;
        while i > 0 {
            i -= 1;
            let bit = ((value >> i) & 1) as u8;
            self.write_bit(bit);
        }
    }

    /// Emit the current byte and reset the window (honouring 0xFF stuffing).
    fn flush_byte(&mut self) {
        let byte = self.cur;
        self.out.push(byte);
        self.last_was_ff = byte == 0xFF;
        self.cur = 0;
        self.nbits = 0;
        self.capacity = if self.last_was_ff { 7 } else { 8 };
    }

    /// Pad with zero bits up to the next byte boundary and return all bytes.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.flush_byte();
        }
        self.out
    }

    /// Number of bits written so far (for diagnostics).
    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.out.len() * 8 + usize::from(self.nbits)
    }
}

/// Encode the unsigned `value` as an `lblock = 3` length field (matching the
/// decoder's `decode_block_length`): emit the single-pass indicator (`0`), then
/// `extra` `1`-bits, a `0` terminator, then `value` in `3 + extra` bits.
fn write_block_length(writer: &mut J2kBitWriter, value: usize) -> Jp2Result<()> {
    // Single coding pass for the single-layer lossless case: pass indicator 0.
    writer.write_bit(0);

    let lblock: u32 = 3;
    // Choose the minimal number of extra bits so `value` fits in lblock+extra.
    let mut total_bits = lblock;
    while (1u64 << total_bits) <= value as u64 {
        total_bits += 1;
        if total_bits > 30 {
            return Err(Jp2Error::InternalError(
                "Tier-2 block length exceeds 30 bits".to_string(),
            ));
        }
    }
    let extra = total_bits - lblock;
    for _ in 0..extra {
        writer.write_bit(1);
    }
    writer.write_bit(0);
    writer.write_bits(value as u32, total_bits as u8);
    Ok(())
}

/// Assemble one Tier-2 packet for a tile-component from per-code-block streams.
///
/// `block_streams[i]` is the MQ-compressed byte data for code-block `i` in the
/// decoder's scan order (LL first, then HL/LH/HH per level coarsest→finest, each
/// subband scanned block-row major). An empty stream (`len == 0`) marks a block
/// excluded from the packet (all coefficients zero). The returned bytes are the
/// complete tile data: byte-aligned packet header followed by the included block
/// bodies in order.
pub fn assemble_packet(block_streams: &[Vec<u8>]) -> Jp2Result<Vec<u8>> {
    let any_included = block_streams.iter().any(|b| !b.is_empty());

    let mut writer = J2kBitWriter::new();

    if !any_included {
        // Empty packet: a single 0 bit (the decoder returns all blocks excluded).
        writer.write_bit(0);
        return Ok(writer.finish());
    }

    // Non-empty packet flag.
    writer.write_bit(1);

    // First sub-header: inclusion bit + zero-bit-plane count per block.
    for stream in block_streams {
        let included = !stream.is_empty();
        writer.write_bit(u8::from(included));
        if included {
            // Zero bit-planes coded as unary: we use 0 → a single terminating 0
            // bit. (The decoder consumes 1-bits until a 0; `num_bit_planes` is the
            // full component bit depth, so no leading zero bit-planes are signalled.)
            writer.write_bit(0);
        }
    }

    // Second sub-header: code-block byte lengths for included blocks.
    for stream in block_streams {
        if !stream.is_empty() {
            write_block_length(&mut writer, stream.len())?;
        }
    }

    // Byte-align the header, then append the included block bodies in order.
    let mut tile_data = writer.finish();
    for stream in block_streams {
        if !stream.is_empty() {
            tile_data.extend_from_slice(stream);
        }
    }

    Ok(tile_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg2000::bitreader::J2kBitReader;
    use crate::jpeg2000::tier2::parse_packet_header;

    #[test]
    fn bitwriter_roundtrip_simple() {
        let mut w = J2kBitWriter::new();
        w.write_bit(1);
        w.write_bit(0);
        w.write_bits(0b1011, 4);
        w.write_bits(0xAB, 8);
        let bytes = w.finish();
        let mut r = J2kBitReader::new(&bytes);
        assert_eq!(r.read_bit().expect("bit"), 1);
        assert_eq!(r.read_bit().expect("bit"), 0);
        assert_eq!(r.read_bits(4).expect("bits"), 0b1011);
        assert_eq!(r.read_bits(8).expect("bits"), 0xAB);
    }

    #[test]
    fn bitwriter_stuffing_after_ff() {
        // Force a 0xFF byte then more bits; the reader must recover them.
        let mut w = J2kBitWriter::new();
        w.write_bits(0xFF, 8);
        w.write_bits(0b101, 3);
        let bytes = w.finish();
        // After a 0xFF, the next byte's top bit is stuffed 0, so byte must be <= 0x7F.
        assert_eq!(bytes[0], 0xFF);
        assert!(bytes[1] <= 0x7F, "stuffed byte must have top bit 0");
        let mut r = J2kBitReader::new(&bytes);
        assert_eq!(r.read_bits(8).expect("bits"), 0xFF);
        assert_eq!(r.read_bits(3).expect("bits"), 0b101);
    }

    #[test]
    fn empty_packet_decodes_all_excluded() {
        let streams: Vec<Vec<u8>> = vec![Vec::new(); 5];
        let data = assemble_packet(&streams).expect("assemble");
        let mut r = J2kBitReader::new(&data);
        let header = parse_packet_header(&mut r, 5).expect("parse");
        assert!(header.included_blocks.iter().all(|&b| !b));
    }

    #[test]
    fn single_included_block_lengths_match() {
        let streams: Vec<Vec<u8>> = vec![vec![1u8, 2, 3], Vec::new(), vec![9u8; 200]];
        let data = assemble_packet(&streams).expect("assemble");
        let mut r = J2kBitReader::new(&data);
        let header = parse_packet_header(&mut r, 3).expect("parse");
        assert_eq!(header.included_blocks, vec![true, false, true]);
        assert_eq!(header.data_lengths[0], 3);
        assert_eq!(header.data_lengths[2], 200);
    }

    #[test]
    fn lengths_various_sizes() {
        for &len in &[1usize, 7, 8, 15, 16, 100, 255, 256, 1000, 65535] {
            let streams = vec![vec![0u8; len]];
            let data = assemble_packet(&streams).expect("assemble");
            let mut r = J2kBitReader::new(&data);
            let header = parse_packet_header(&mut r, 1).expect("parse");
            assert!(header.included_blocks[0]);
            assert_eq!(header.data_lengths[0], len, "length {len} round-trip");
        }
    }
}
