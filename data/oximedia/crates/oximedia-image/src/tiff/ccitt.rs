//! CCITT fax compression (ITU-T T.4 / T.6) for bilevel TIFF imagery.
//!
//! Implements:
//! - `CcittRle` (TIFF compression 2): modified Huffman 1D run-length.
//! - `CcittFax3` (3): Group 3 — 1D modified Huffman, with EOL byte alignment.
//! - `CcittFax4` (4): Group 4 / MMR — pure 2D (T.6).
//!
//! All three share the T.4 terminating + make-up run-length code tables
//! (white and black) and the T.6 vertical/pass/horizontal 2D mode codes.
//!
//! Only bilevel (1-bit) imagery is supported. The decoded output is a packed
//! 1-bit-per-pixel bitmap, MSB-first, each row padded to a whole byte — the
//! same layout an uncompressed 1-bit TIFF strip would carry.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use crate::error::{ImageError, ImageResult};

// ===========================================================================
// T.4 run-length code tables (ITU-T T.4 Tables 1, 2, 3)
// ===========================================================================
//
// Each entry is `(bit_length, code_word, run_length)`.
// Terminating codes encode runs 0..=63. Make-up codes encode multiples of 64.
// The "extended" make-up codes (T.4 Table 3) are colour-independent and
// encode runs 1792..=2560.

/// White terminating run-length codes (run 0..=63).
const WHITE_TERMINATING: [(u8, u16, u16); 64] = [
    (8, 0x35, 0),
    (6, 0x07, 1),
    (4, 0x07, 2),
    (4, 0x08, 3),
    (4, 0x0B, 4),
    (4, 0x0C, 5),
    (4, 0x0E, 6),
    (4, 0x0F, 7),
    (5, 0x13, 8),
    (5, 0x14, 9),
    (5, 0x07, 10),
    (5, 0x08, 11),
    (6, 0x08, 12),
    (6, 0x03, 13),
    (6, 0x34, 14),
    (6, 0x35, 15),
    (6, 0x2A, 16),
    (6, 0x2B, 17),
    (7, 0x27, 18),
    (7, 0x0C, 19),
    (7, 0x08, 20),
    (7, 0x17, 21),
    (7, 0x03, 22),
    (7, 0x04, 23),
    (7, 0x28, 24),
    (7, 0x2B, 25),
    (7, 0x13, 26),
    (7, 0x24, 27),
    (7, 0x18, 28),
    (8, 0x02, 29),
    (8, 0x03, 30),
    (8, 0x1A, 31),
    (8, 0x1B, 32),
    (8, 0x12, 33),
    (8, 0x13, 34),
    (8, 0x14, 35),
    (8, 0x15, 36),
    (8, 0x16, 37),
    (8, 0x17, 38),
    (8, 0x28, 39),
    (8, 0x29, 40),
    (8, 0x2A, 41),
    (8, 0x2B, 42),
    (8, 0x2C, 43),
    (8, 0x2D, 44),
    (8, 0x04, 45),
    (8, 0x05, 46),
    (8, 0x0A, 47),
    (8, 0x0B, 48),
    (8, 0x52, 49),
    (8, 0x53, 50),
    (8, 0x54, 51),
    (8, 0x55, 52),
    (8, 0x24, 53),
    (8, 0x25, 54),
    (8, 0x58, 55),
    (8, 0x59, 56),
    (8, 0x5A, 57),
    (8, 0x5B, 58),
    (8, 0x4A, 59),
    (8, 0x4B, 60),
    (8, 0x32, 61),
    (8, 0x33, 62),
    (8, 0x34, 63),
];

/// White make-up codes (run = multiple of 64, 64..=1728).
const WHITE_MAKEUP: [(u8, u16, u16); 27] = [
    (5, 0x1B, 64),
    (5, 0x12, 128),
    (6, 0x17, 192),
    (7, 0x37, 256),
    (8, 0x36, 320),
    (8, 0x37, 384),
    (8, 0x64, 448),
    (8, 0x65, 512),
    (8, 0x68, 576),
    (8, 0x67, 640),
    (9, 0xCC, 704),
    (9, 0xCD, 768),
    (9, 0xD2, 832),
    (9, 0xD3, 896),
    (9, 0xD4, 960),
    (9, 0xD5, 1024),
    (9, 0xD6, 1088),
    (9, 0xD7, 1152),
    (9, 0xD8, 1216),
    (9, 0xD9, 1280),
    (9, 0xDA, 1344),
    (9, 0xDB, 1408),
    (9, 0x98, 1472),
    (9, 0x99, 1536),
    (9, 0x9A, 1600),
    (6, 0x18, 1664),
    (9, 0x9B, 1728),
];

/// Black terminating run-length codes (run 0..=63).
const BLACK_TERMINATING: [(u8, u16, u16); 64] = [
    (10, 0x37, 0),
    (3, 0x02, 1),
    (2, 0x03, 2),
    (2, 0x02, 3),
    (3, 0x03, 4),
    (4, 0x03, 5),
    (4, 0x02, 6),
    (5, 0x03, 7),
    (6, 0x05, 8),
    (6, 0x04, 9),
    (7, 0x04, 10),
    (7, 0x05, 11),
    (7, 0x07, 12),
    (8, 0x04, 13),
    (8, 0x07, 14),
    (9, 0x18, 15),
    (10, 0x17, 16),
    (10, 0x18, 17),
    (10, 0x08, 18),
    (11, 0x67, 19),
    (11, 0x68, 20),
    (11, 0x6C, 21),
    (11, 0x37, 22),
    (11, 0x28, 23),
    (11, 0x17, 24),
    (11, 0x18, 25),
    (12, 0xCA, 26),
    (12, 0xCB, 27),
    (12, 0xCC, 28),
    (12, 0xCD, 29),
    (12, 0x68, 30),
    (12, 0x69, 31),
    (12, 0x6A, 32),
    (12, 0x6B, 33),
    (12, 0xD2, 34),
    (12, 0xD3, 35),
    (12, 0xD4, 36),
    (12, 0xD5, 37),
    (12, 0xD6, 38),
    (12, 0xD7, 39),
    (12, 0x6C, 40),
    (12, 0x6D, 41),
    (12, 0xDA, 42),
    (12, 0xDB, 43),
    (12, 0x54, 44),
    (12, 0x55, 45),
    (12, 0x56, 46),
    (12, 0x57, 47),
    (12, 0x64, 48),
    (12, 0x65, 49),
    (12, 0x52, 50),
    (12, 0x53, 51),
    (12, 0x24, 52),
    (12, 0x37, 53),
    (12, 0x38, 54),
    (12, 0x27, 55),
    (12, 0x28, 56),
    (12, 0x58, 57),
    (12, 0x59, 58),
    (12, 0x2B, 59),
    (12, 0x2C, 60),
    (12, 0x5A, 61),
    (12, 0x66, 62),
    (12, 0x67, 63),
];

/// Black make-up codes (run = multiple of 64, 64..=1728).
const BLACK_MAKEUP: [(u8, u16, u16); 27] = [
    (10, 0x0F, 64),
    (12, 0xC8, 128),
    (12, 0xC9, 192),
    (12, 0x5B, 256),
    (12, 0x33, 320),
    (12, 0x34, 384),
    (12, 0x35, 448),
    (13, 0x6C, 512),
    (13, 0x6D, 576),
    (13, 0x4A, 640),
    (13, 0x4B, 704),
    (13, 0x4C, 768),
    (13, 0x4D, 832),
    (13, 0x72, 896),
    (13, 0x73, 960),
    (13, 0x74, 1024),
    (13, 0x75, 1088),
    (13, 0x76, 1152),
    (13, 0x77, 1216),
    (13, 0x52, 1280),
    (13, 0x53, 1344),
    (13, 0x54, 1408),
    (13, 0x55, 1472),
    (13, 0x5A, 1536),
    (13, 0x5B, 1600),
    (13, 0x64, 1664),
    (13, 0x65, 1728),
];

/// Extended (colour-independent) make-up codes, runs 1792..=2560.
const EXTENDED_MAKEUP: [(u8, u16, u16); 13] = [
    (11, 0x08, 1792),
    (11, 0x0C, 1856),
    (11, 0x0D, 1920),
    (12, 0x12, 1984),
    (12, 0x13, 2048),
    (12, 0x14, 2112),
    (12, 0x15, 2176),
    (12, 0x16, 2240),
    (12, 0x17, 2304),
    (12, 0x1C, 2368),
    (12, 0x1D, 2432),
    (12, 0x1E, 2496),
    (12, 0x1F, 2560),
];

/// EOL (end-of-line) code: 11 zero bits then a 1 (`000000000001`, 12 bits).
const EOL_LEN: u8 = 12;
const EOL_CODE: u16 = 0x001;

// ===========================================================================
// Bit reader (MSB-first)
// ===========================================================================

struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit position from the start of `data`.
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    fn total_bits(&self) -> usize {
        self.data.len() * 8
    }

    fn exhausted(&self) -> bool {
        self.bit_pos >= self.total_bits()
    }

    /// Read a single bit, MSB-first. Returns `None` past the end of data.
    fn read_bit(&mut self) -> Option<u8> {
        if self.bit_pos >= self.total_bits() {
            return None;
        }
        let byte = self.data[self.bit_pos >> 3];
        let bit = (byte >> (7 - (self.bit_pos & 7))) & 1;
        self.bit_pos += 1;
        Some(bit)
    }

    /// Peek up to `n` bits without consuming (zero-padded past end).
    fn peek(&self, n: u8) -> u32 {
        let mut value = 0u32;
        for i in 0..n {
            let pos = self.bit_pos + i as usize;
            let bit = if pos < self.total_bits() {
                let byte = self.data[pos >> 3];
                u32::from((byte >> (7 - (pos & 7))) & 1)
            } else {
                0
            };
            value = (value << 1) | bit;
        }
        value
    }

    fn skip(&mut self, n: u8) {
        self.bit_pos += n as usize;
    }

    /// Advance to the next byte boundary.
    fn align_to_byte(&mut self) {
        let rem = self.bit_pos & 7;
        if rem != 0 {
            self.bit_pos += 8 - rem;
        }
    }
}

// ===========================================================================
// Bit writer (MSB-first)
// ===========================================================================

struct BitWriter {
    buf: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            cur: 0,
            nbits: 0,
        }
    }

    fn write_bit(&mut self, bit: u8) {
        self.cur = (self.cur << 1) | (bit & 1);
        self.nbits += 1;
        if self.nbits == 8 {
            self.buf.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    /// Write `len` low bits of `code`, MSB-first.
    fn write_code(&mut self, len: u8, code: u16) {
        for i in (0..len).rev() {
            self.write_bit(((code >> i) & 1) as u8);
        }
    }

    fn align_to_byte(&mut self) {
        if self.nbits != 0 {
            self.cur <<= 8 - self.nbits;
            self.buf.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    fn into_bytes(mut self) -> Vec<u8> {
        self.align_to_byte();
        self.buf
    }

    fn bit_count(&self) -> usize {
        self.buf.len() * 8 + self.nbits as usize
    }
}

// ===========================================================================
// Run-length code lookup (decode side)
// ===========================================================================

/// Decode a single run-length code (terminating or make-up) for the given
/// colour. Returns `(run, is_terminating)`, or `None` if no valid prefix is
/// found. A make-up run (>= 64) means "keep reading".
fn decode_run_code(reader: &mut BitReader, white: bool) -> Option<(u16, bool)> {
    let term = if white {
        &WHITE_TERMINATING[..]
    } else {
        &BLACK_TERMINATING[..]
    };
    let makeup = if white {
        &WHITE_MAKEUP[..]
    } else {
        &BLACK_MAKEUP[..]
    };

    // Codes range from 2 to 14 bits. EOL is handled separately by the caller.
    for len in 2u8..=14 {
        let bits = reader.peek(len);
        for &(clen, code, run) in term {
            if clen == len && u32::from(code) == bits {
                reader.skip(len);
                return Some((run, true));
            }
        }
        for &(clen, code, run) in makeup {
            if clen == len && u32::from(code) == bits {
                reader.skip(len);
                return Some((run, false));
            }
        }
        for &(clen, code, run) in &EXTENDED_MAKEUP {
            if clen == len && u32::from(code) == bits {
                reader.skip(len);
                return Some((run, false));
            }
        }
    }
    None
}

/// Decode a complete run for one colour: accumulate make-up codes until a
/// terminating code closes the run.
fn decode_full_run(reader: &mut BitReader, white: bool) -> Option<u32> {
    let mut total = 0u32;
    loop {
        let (run, is_term) = decode_run_code(reader, white)?;
        total += u32::from(run);
        if is_term {
            return Some(total);
        }
        if total > 1 << 24 {
            return None; // runaway guard
        }
    }
}

// ===========================================================================
// Run-length code emission (encode side)
// ===========================================================================

/// Emit a run of `run` pixels of the given colour using make-up + terminating
/// codes per T.4.
fn emit_run(writer: &mut BitWriter, mut run: u32, white: bool) {
    let term = if white {
        &WHITE_TERMINATING[..]
    } else {
        &BLACK_TERMINATING[..]
    };
    let makeup = if white {
        &WHITE_MAKEUP[..]
    } else {
        &BLACK_MAKEUP[..]
    };

    // Extended make-up codes cover 1792..=2560 (steps of 64).
    while run >= 2560 {
        let (len, code, _) = EXTENDED_MAKEUP[EXTENDED_MAKEUP.len() - 1];
        writer.write_code(len, code);
        run -= 2560;
    }
    if run >= 1792 {
        let mut chosen = EXTENDED_MAKEUP[0];
        for &e in &EXTENDED_MAKEUP {
            if u32::from(e.2) <= run {
                chosen = e;
            }
        }
        writer.write_code(chosen.0, chosen.1);
        run -= u32::from(chosen.2);
    }
    // Ordinary make-up codes cover multiples of 64 up to 1728.
    while run >= 64 {
        let mut chosen = makeup[0];
        for &m in makeup {
            if u32::from(m.2) <= run {
                chosen = m;
            }
        }
        writer.write_code(chosen.0, chosen.1);
        run -= u32::from(chosen.2);
    }
    // Terminating code for the remaining 0..=63 pixels.
    let (len, code, _) = term[run as usize];
    writer.write_code(len, code);
}

// ===========================================================================
// 2D mode codes (ITU-T T.4 Table 4 / T.6)
// ===========================================================================
//
// Mode codes, prefix-free:
//   Pass         0001            (4 bits)
//   Horizontal   001             (3 bits)
//   V0           1               (1 bit)
//   VR1          011             (3 bits)
//   VR2          000011          (6 bits)
//   VR3          0000011         (7 bits)
//   VL1          010             (3 bits)
//   VL2          000010          (6 bits)
//   VL3          0000010         (7 bits)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode2d {
    Pass,
    Horizontal,
    Vertical(i32),
}

/// Decode a 2D mode code from the bit reader.
fn decode_mode(reader: &mut BitReader) -> Option<Mode2d> {
    if reader.peek(1) == 0b1 {
        reader.skip(1);
        return Some(Mode2d::Vertical(0));
    }
    let p3 = reader.peek(3);
    match p3 {
        0b011 => {
            reader.skip(3);
            return Some(Mode2d::Vertical(1));
        }
        0b010 => {
            reader.skip(3);
            return Some(Mode2d::Vertical(-1));
        }
        0b001 => {
            reader.skip(3);
            return Some(Mode2d::Horizontal);
        }
        _ => {}
    }
    if reader.peek(4) == 0b0001 {
        reader.skip(4);
        return Some(Mode2d::Pass);
    }
    let p6 = reader.peek(6);
    match p6 {
        0b000011 => {
            reader.skip(6);
            return Some(Mode2d::Vertical(2));
        }
        0b000010 => {
            reader.skip(6);
            return Some(Mode2d::Vertical(-2));
        }
        _ => {}
    }
    let p7 = reader.peek(7);
    match p7 {
        0b0000011 => {
            reader.skip(7);
            return Some(Mode2d::Vertical(3));
        }
        0b0000010 => {
            reader.skip(7);
            return Some(Mode2d::Vertical(-3));
        }
        _ => {}
    }
    None
}

/// Emit a 2D mode code.
fn emit_mode(writer: &mut BitWriter, mode: Mode2d) {
    match mode {
        Mode2d::Pass => writer.write_code(4, 0b0001),
        Mode2d::Horizontal => writer.write_code(3, 0b001),
        Mode2d::Vertical(0) => writer.write_code(1, 0b1),
        Mode2d::Vertical(1) => writer.write_code(3, 0b011),
        Mode2d::Vertical(-1) => writer.write_code(3, 0b010),
        Mode2d::Vertical(2) => writer.write_code(6, 0b000011),
        Mode2d::Vertical(-2) => writer.write_code(6, 0b000010),
        Mode2d::Vertical(3) => writer.write_code(7, 0b0000011),
        Mode2d::Vertical(-3) => writer.write_code(7, 0b0000010),
        Mode2d::Vertical(_) => {
            // Should never happen — callers clamp deltas to +-3.
            writer.write_code(1, 0b1);
        }
    }
}

// ===========================================================================
// Row representation as changing-element lists
// ===========================================================================
//
// A bilevel row is represented as a sorted list of "changing elements": the
// pixel positions where the colour changes. The image is conceptually padded
// with an imaginary white pixel just left of column 0. A row that is all-white
// has an empty change list.

/// Convert an unpacked row (`true` = black) to a changing-element list.
fn row_to_changes(pixels: &[bool]) -> Vec<usize> {
    let mut changes = Vec::new();
    let mut cur = false; // imaginary pixel left of column 0 is white
    for (x, &p) in pixels.iter().enumerate() {
        if p != cur {
            changes.push(x);
            cur = p;
        }
    }
    changes
}

/// Reconstruct a packed row from a changing-element list.
fn changes_to_row(changes: &[usize], width: usize) -> Vec<bool> {
    let mut row = vec![false; width];
    let mut cur = false;
    let mut x = 0usize;
    for &c in changes {
        let c = c.min(width);
        if cur {
            for cell in row.iter_mut().take(c).skip(x) {
                *cell = true;
            }
        }
        x = c;
        cur = !cur;
    }
    if cur {
        for cell in row.iter_mut().skip(x) {
            *cell = true;
        }
    }
    row
}

/// Colour of the pixel just *before* position `pos` in a row described by its
/// change list. Position 0 is preceded by the imaginary white pixel.
fn color_at(changes: &[usize], pos: usize) -> bool {
    let n = changes.iter().take_while(|&&c| c < pos).count();
    n % 2 == 1
}

/// Find `b1`: the first changing element on the reference line strictly to the
/// right of `a0` whose run colour is opposite to `a0`'s colour.
fn find_b1(ref_changes: &[usize], a0: isize, a0_color: bool, width: usize) -> usize {
    // The run *starting* at ref_changes[i] has colour `i % 2 == 0` -> black.
    for (i, &c) in ref_changes.iter().enumerate() {
        if (c as isize) > a0 {
            let starting_color = i % 2 == 0;
            if starting_color != a0_color {
                return c;
            }
        }
    }
    width
}

/// Find `b2`: the next changing element on the reference line after `b1`.
fn find_b2(ref_changes: &[usize], b1: usize, width: usize) -> usize {
    for &c in ref_changes {
        if c > b1 {
            return c;
        }
    }
    width
}

/// Next changing element strictly right of `pos`, or `width` if none.
fn next_change_after(changes: &[usize], pos: isize, width: usize) -> usize {
    for &c in changes {
        if (c as isize) > pos {
            return c;
        }
    }
    width
}

// ===========================================================================
// Pixel (un)packing
// ===========================================================================

/// Unpack a packed 1-bpp strip into rows of `bool` (`true` = black).
///
/// `white_is_zero` selects polarity: with `WhiteIsZero` photometric a 0 bit is
/// white, so a 1 bit is black; with `BlackIsZero` the meaning is inverted.
fn unpack_strip(data: &[u8], width: usize, rows: usize, white_is_zero: bool) -> Vec<Vec<bool>> {
    let row_bytes = width.div_ceil(8);
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = vec![false; width];
        let base = r * row_bytes;
        for (x, cell) in row.iter_mut().enumerate() {
            let byte_idx = base + (x >> 3);
            let bit = if byte_idx < data.len() {
                (data[byte_idx] >> (7 - (x & 7))) & 1
            } else {
                0
            };
            *cell = if white_is_zero { bit == 1 } else { bit == 0 };
        }
        out.push(row);
    }
    out
}

/// Pack rows of `bool` back into a packed 1-bpp strip.
fn pack_rows(rows: &[Vec<bool>], width: usize, white_is_zero: bool) -> Vec<u8> {
    let row_bytes = width.div_ceil(8);
    let mut out = vec![0u8; rows.len() * row_bytes];
    for (r, row) in rows.iter().enumerate() {
        let base = r * row_bytes;
        for (x, &black) in row.iter().enumerate() {
            let bit = if white_is_zero {
                u8::from(black)
            } else {
                u8::from(!black)
            };
            if bit != 0 {
                out[base + (x >> 3)] |= 1 << (7 - (x & 7));
            }
        }
    }
    out
}

// ===========================================================================
// 1D row decode (modified Huffman) — shared by RLE and G3-1D
// ===========================================================================

/// Decode one 1D-coded scan line into a row of pixels.
///
/// A 1D line is a sequence of alternating white/black runs starting with white.
fn decode_1d_row(reader: &mut BitReader, width: usize) -> ImageResult<Vec<bool>> {
    let mut row = vec![false; width];
    let mut x = 0usize;
    let mut white = true;
    while x < width {
        let run = decode_full_run(reader, white)
            .ok_or_else(|| ImageError::compression("CCITT: invalid 1D run-length code"))?;
        let end = (x + run as usize).min(width);
        if !white {
            for cell in row.iter_mut().take(end).skip(x) {
                *cell = true;
            }
        }
        x = end;
        white = !white;
    }
    Ok(row)
}

/// Encode one row in 1D modified-Huffman form (alternating white/black runs).
fn encode_1d_row(writer: &mut BitWriter, row: &[bool]) {
    let mut x = 0usize;
    let mut white = true;
    let width = row.len();
    while x < width {
        let mut run = 0u32;
        while x < width && row[x] == !white {
            run += 1;
            x += 1;
        }
        emit_run(writer, run, white);
        white = !white;
    }
}

// ===========================================================================
// 2D row decode / encode (T.6 / MMR)
// ===========================================================================

/// Decode one 2D-coded scan line given the reference line's change list.
fn decode_2d_row(
    reader: &mut BitReader,
    ref_changes: &[usize],
    width: usize,
) -> ImageResult<Vec<usize>> {
    let mut changes: Vec<usize> = Vec::new();
    let mut a0: isize = -1;
    let mut a0_color = false; // imaginary pixel left of column 0 is white

    while (a0 as i64) < width as i64 {
        let b1 = find_b1(ref_changes, a0, a0_color, width);
        let b2 = find_b2(ref_changes, b1, width);

        let mode = decode_mode(reader)
            .ok_or_else(|| ImageError::compression("CCITT: invalid 2D mode code"))?;

        match mode {
            Mode2d::Pass => {
                a0 = b2 as isize;
            }
            Mode2d::Horizontal => {
                let start = if a0 < 0 { 0 } else { a0 as usize };
                let run1 = decode_full_run(reader, !a0_color)
                    .ok_or_else(|| ImageError::compression("CCITT: invalid horizontal run 1"))?;
                let run2 = decode_full_run(reader, a0_color)
                    .ok_or_else(|| ImageError::compression("CCITT: invalid horizontal run 2"))?;
                let a1 = (start + run1 as usize).min(width);
                let a2 = (a1 + run2 as usize).min(width);
                changes.push(a1);
                changes.push(a2);
                a0 = a2 as isize;
            }
            Mode2d::Vertical(delta) => {
                let a1 = (b1 as isize + delta as isize).clamp(0, width as isize) as usize;
                changes.push(a1);
                a0 = a1 as isize;
                a0_color = !a0_color;
            }
        }
        if changes.len() > width + 2 {
            return Err(ImageError::compression("CCITT: row overflow"));
        }
    }
    Ok(changes)
}

/// Encode one row in 2D form (T.6) against a reference line. The encoder
/// greedily prefers pass, then vertical, then horizontal mode.
fn encode_2d_row(
    writer: &mut BitWriter,
    ref_changes: &[usize],
    cur_changes: &[usize],
    width: usize,
) {
    let mut a0: isize = -1;
    let mut a0_color = false;

    loop {
        let a1 = next_change_after(cur_changes, a0, width);
        let a2 = next_change_after(cur_changes, a1 as isize, width);

        let b1 = find_b1(ref_changes, a0, a0_color, width);
        let b2 = find_b2(ref_changes, b1, width);

        if (b2 as isize) < a1 as isize {
            // Pass mode: a1 lies to the right of b2.
            emit_mode(writer, Mode2d::Pass);
            a0 = b2 as isize;
        } else {
            let delta = a1 as isize - b1 as isize;
            if (-3..=3).contains(&delta) {
                emit_mode(writer, Mode2d::Vertical(delta as i32));
                a0 = a1 as isize;
                a0_color = !a0_color;
            } else {
                let start = if a0 < 0 { 0 } else { a0 as usize };
                let run1 = a1.saturating_sub(start) as u32;
                let run2 = a2.saturating_sub(a1) as u32;
                emit_mode(writer, Mode2d::Horizontal);
                emit_run(writer, run1, !a0_color);
                emit_run(writer, run2, a0_color);
                a0 = a2 as isize;
            }
        }
        if a0 as i64 >= width as i64 {
            break;
        }
    }
}

// ===========================================================================
// Public decode entry points
// ===========================================================================

/// Decode a `CcittRle` (TIFF compression 2) strip.
///
/// Modified-Huffman 1D run-length; TIFF compression 2 byte-aligns every row.
pub fn decode_ccitt_rle(
    data: &[u8],
    width: usize,
    rows: usize,
    white_is_zero: bool,
) -> ImageResult<Vec<u8>> {
    if width == 0 {
        return Ok(Vec::new());
    }
    let mut reader = BitReader::new(data);
    let mut out_rows: Vec<Vec<bool>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        if reader.exhausted() {
            out_rows.push(vec![false; width]);
            continue;
        }
        let row = decode_1d_row(&mut reader, width)?;
        out_rows.push(row);
        reader.align_to_byte();
    }
    Ok(pack_rows(&out_rows, width, white_is_zero))
}

/// Decode a `CcittFax3` (TIFF compression 3, ITU-T T.4 Group 3) strip.
///
/// Implements 1D modified-Huffman coding — the default for `CcittFax3` when the
/// `T4Options` 2D-coding bit is absent (the overwhelmingly common case for
/// TIFF compression 3). EOL codes (`000000000001`) delimit lines and any
/// fill-bit padding before an EOL is absorbed. Streams that omit EOLs entirely
/// are also handled: lines decode back-to-back.
pub fn decode_ccitt_fax3(
    data: &[u8],
    width: usize,
    rows: usize,
    white_is_zero: bool,
) -> ImageResult<Vec<u8>> {
    if width == 0 {
        return Ok(Vec::new());
    }
    let mut reader = BitReader::new(data);
    let mut out_rows: Vec<Vec<bool>> = Vec::with_capacity(rows);

    for _ in 0..rows {
        skip_fill_and_eol(&mut reader);
        if reader.exhausted() {
            out_rows.push(vec![false; width]);
            continue;
        }
        let row = decode_1d_row(&mut reader, width)?;
        out_rows.push(row);
    }
    Ok(pack_rows(&out_rows, width, white_is_zero))
}

/// Absorb fill-bit padding and one or more EOL codes at the reader's cursor.
///
/// T.4 permits a run of "fill" zero bits immediately before an EOL. An EOL is
/// `00000000 0001`. If `>= 11` zero bits followed by a 1 are reachable, they
/// form an EOL and are consumed; consecutive EOLs (RTC / EOFB) are all skipped.
fn skip_fill_and_eol(reader: &mut BitReader) {
    loop {
        let mut zeros = 0u8;
        while zeros < 64 {
            let pos = reader.bit_pos + zeros as usize;
            if pos >= reader.total_bits() {
                break;
            }
            let byte = reader.data[pos >> 3];
            let bit = (byte >> (7 - (pos & 7))) & 1;
            if bit == 1 {
                break;
            }
            zeros += 1;
        }
        // Fewer than 11 leading zeros => real run-length data; stop skipping.
        if zeros >= 11 {
            let one_pos = reader.bit_pos + zeros as usize;
            if one_pos < reader.total_bits() {
                reader.skip(zeros);
                reader.skip(1);
                continue;
            }
        }
        break;
    }
}

/// Decode a `CcittFax4` (TIFF compression 4, ITU-T T.6 / MMR) strip.
///
/// Pure 2D coding: every line is coded against the previous line; the first
/// line is coded against an imaginary all-white line. An EOFB (two EOLs) may
/// terminate the stream.
pub fn decode_ccitt_fax4(
    data: &[u8],
    width: usize,
    rows: usize,
    white_is_zero: bool,
) -> ImageResult<Vec<u8>> {
    if width == 0 {
        return Ok(Vec::new());
    }
    let mut reader = BitReader::new(data);
    let mut out_rows: Vec<Vec<bool>> = Vec::with_capacity(rows);
    let mut ref_changes: Vec<usize> = Vec::new();

    for _ in 0..rows {
        if reader.exhausted() || peek_is_eofb(&reader) {
            out_rows.push(vec![false; width]);
            ref_changes.clear();
            continue;
        }
        let changes = decode_2d_row(&mut reader, &ref_changes, width)?;
        let row = changes_to_row(&changes, width);
        ref_changes = changes;
        out_rows.push(row);
    }
    Ok(pack_rows(&out_rows, width, white_is_zero))
}

/// Returns `true` if the reader is at an EOFB (two consecutive EOLs).
fn peek_is_eofb(reader: &BitReader) -> bool {
    reader.peek(24) == ((u32::from(EOL_CODE) << 12) | u32::from(EOL_CODE))
}

// ===========================================================================
// Public encode entry points
// ===========================================================================

/// Encode a packed 1-bpp strip as `CcittRle` (TIFF compression 2).
pub fn encode_ccitt_rle(data: &[u8], width: usize, rows: usize, white_is_zero: bool) -> Vec<u8> {
    let pixel_rows = unpack_strip(data, width, rows, white_is_zero);
    let mut writer = BitWriter::new();
    for row in &pixel_rows {
        encode_1d_row(&mut writer, row);
        // TIFF compression 2 byte-aligns each row.
        writer.align_to_byte();
    }
    writer.into_bytes()
}

/// Encode a packed 1-bpp strip as `CcittFax3` (TIFF compression 3, 1D coding).
///
/// Each line is preceded by an EOL code, as ITU-T T.4 prescribes for Group 3.
pub fn encode_ccitt_fax3(data: &[u8], width: usize, rows: usize, white_is_zero: bool) -> Vec<u8> {
    let pixel_rows = unpack_strip(data, width, rows, white_is_zero);
    let mut writer = BitWriter::new();
    writer.write_code(EOL_LEN, EOL_CODE); // leading EOL
    for row in &pixel_rows {
        encode_1d_row(&mut writer, row);
        writer.write_code(EOL_LEN, EOL_CODE); // EOL terminates every line
    }
    writer.into_bytes()
}

/// Encode a packed 1-bpp strip as `CcittFax4` (TIFF compression 4, T.6 / MMR).
///
/// Pure 2D coding; the encoder greedily picks pass / vertical / horizontal mode
/// against the reference line for every coding line.
pub fn encode_ccitt_fax4(data: &[u8], width: usize, rows: usize, white_is_zero: bool) -> Vec<u8> {
    let pixel_rows = unpack_strip(data, width, rows, white_is_zero);
    let mut writer = BitWriter::new();
    let mut ref_changes: Vec<usize> = Vec::new();
    for row in &pixel_rows {
        let cur_changes = row_to_changes(row);
        encode_2d_row(&mut writer, &ref_changes, &cur_changes, width);
        ref_changes = cur_changes;
    }
    // EOFB: two EOL codes terminate a T.6 block.
    writer.write_code(EOL_LEN, EOL_CODE);
    writer.write_code(EOL_LEN, EOL_CODE);
    writer.into_bytes()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_changes_roundtrip() {
        let row = [false, false, true, true, true, false, true, false];
        let changes = row_to_changes(&row);
        assert_eq!(changes, vec![2, 5, 6, 7]);
        let back = changes_to_row(&changes, row.len());
        assert_eq!(back, row);
    }

    #[test]
    fn test_changes_all_white() {
        let row = [false; 16];
        let changes = row_to_changes(&row);
        assert!(changes.is_empty());
        assert_eq!(changes_to_row(&changes, 16), row.to_vec());
    }

    #[test]
    fn test_changes_all_black() {
        let row = [true; 12];
        let changes = row_to_changes(&row);
        assert_eq!(changes, vec![0]);
        assert_eq!(changes_to_row(&changes, 12), row.to_vec());
    }

    #[test]
    fn test_color_at() {
        let changes = [2usize, 5, 6, 7];
        assert!(!color_at(&changes, 0));
        assert!(!color_at(&changes, 2));
        assert!(color_at(&changes, 3));
        assert!(color_at(&changes, 5));
        assert!(!color_at(&changes, 6));
    }

    #[test]
    fn test_emit_decode_run_white() {
        for run in [0u32, 1, 5, 63, 64, 100, 1000, 1728, 1791, 2560, 5000] {
            let mut w = BitWriter::new();
            emit_run(&mut w, run, true);
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let decoded = decode_full_run(&mut r, true).expect("decode white run");
            assert_eq!(decoded, run, "white run {run}");
        }
    }

    #[test]
    fn test_emit_decode_run_black() {
        for run in [0u32, 1, 7, 63, 64, 128, 999, 1728, 2000, 4096] {
            let mut w = BitWriter::new();
            emit_run(&mut w, run, false);
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let decoded = decode_full_run(&mut r, false).expect("decode black run");
            assert_eq!(decoded, run, "black run {run}");
        }
    }

    #[test]
    fn test_rle_roundtrip_simple() {
        let row0: &[bool] = &[false, false, true, true, true, true, false, false];
        let row1: &[bool] = &[true, false, true, false, true, false, true, false];
        let packed = pack_rows(&[row0.to_vec(), row1.to_vec()], 8, true);
        let enc = encode_ccitt_rle(&packed, 8, 2, true);
        let dec = decode_ccitt_rle(&enc, 8, 2, true).expect("rle decode");
        assert_eq!(dec, packed);
    }

    #[test]
    fn test_rle_roundtrip_wide() {
        let mut row0 = vec![false; 300];
        for cell in row0.iter_mut().take(250).skip(50) {
            *cell = true;
        }
        let mut row1 = vec![true; 300];
        for cell in row1.iter_mut().take(200).skip(100) {
            *cell = false;
        }
        let packed = pack_rows(&[row0, row1], 300, true);
        let enc = encode_ccitt_rle(&packed, 300, 2, true);
        let dec = decode_ccitt_rle(&enc, 300, 2, true).expect("rle decode wide");
        assert_eq!(dec, packed);
    }

    #[test]
    fn test_fax3_1d_roundtrip() {
        let row0: &[bool] = &[false, true, true, false, false, true, false, true];
        let row1: &[bool] = &[true, true, true, true, false, false, false, false];
        let row2: &[bool] = &[false; 8];
        let packed = pack_rows(&[row0.to_vec(), row1.to_vec(), row2.to_vec()], 8, true);
        let enc = encode_ccitt_fax3(&packed, 8, 3, true);
        let dec = decode_ccitt_fax3(&enc, 8, 3, true).expect("fax3 decode");
        assert_eq!(dec, packed);
    }

    #[test]
    fn test_fax3_1d_roundtrip_wide() {
        let mut rows: Vec<Vec<bool>> = Vec::new();
        for r in 0..10 {
            let mut row = vec![false; 200];
            for (x, cell) in row.iter_mut().enumerate() {
                *cell = (x + r) % 7 < 3;
            }
            rows.push(row);
        }
        let packed = pack_rows(&rows, 200, true);
        let enc = encode_ccitt_fax3(&packed, 200, 10, true);
        let dec = decode_ccitt_fax3(&enc, 200, 10, true).expect("fax3 decode wide");
        assert_eq!(dec, packed);
    }

    #[test]
    fn test_fax4_roundtrip_simple() {
        let row0: &[bool] = &[false, false, true, true, false, false, true, true];
        let row1: &[bool] = &[false, true, true, true, true, false, false, true];
        let row2: &[bool] = &[true, true, false, false, false, false, true, true];
        let packed = pack_rows(&[row0.to_vec(), row1.to_vec(), row2.to_vec()], 8, true);
        let enc = encode_ccitt_fax4(&packed, 8, 3, true);
        let dec = decode_ccitt_fax4(&enc, 8, 3, true).expect("fax4 decode");
        assert_eq!(dec, packed);
    }

    #[test]
    fn test_fax4_roundtrip_pattern() {
        // A diagonal pattern exercises pass / vertical / horizontal modes.
        let width = 64;
        let height = 48;
        let mut rows: Vec<Vec<bool>> = Vec::new();
        for r in 0..height {
            let mut row = vec![false; width];
            for (x, cell) in row.iter_mut().enumerate() {
                *cell = ((x as i32 - r as i32).rem_euclid(16)) < 8;
            }
            rows.push(row);
        }
        let packed = pack_rows(&rows, width, true);
        let enc = encode_ccitt_fax4(&packed, width, height, true);
        let dec = decode_ccitt_fax4(&enc, width, height, true).expect("fax4 decode pattern");
        assert_eq!(dec, packed);
    }

    #[test]
    fn test_fax4_roundtrip_all_white() {
        let packed = pack_rows(&vec![vec![false; 32]; 8], 32, true);
        let enc = encode_ccitt_fax4(&packed, 32, 8, true);
        let dec = decode_ccitt_fax4(&enc, 32, 8, true).expect("fax4 white");
        assert_eq!(dec, packed);
    }

    #[test]
    fn test_fax4_roundtrip_all_black() {
        let packed = pack_rows(&vec![vec![true; 32]; 8], 32, true);
        let enc = encode_ccitt_fax4(&packed, 32, 8, true);
        let dec = decode_ccitt_fax4(&enc, 32, 8, true).expect("fax4 black");
        assert_eq!(dec, packed);
    }

    #[test]
    fn test_fax4_compression_ratio() {
        // Mostly-white 256x256 should compress dramatically.
        let width = 256;
        let height = 256;
        let mut rows: Vec<Vec<bool>> = vec![vec![false; width]; height];
        for cell in rows[128].iter_mut() {
            *cell = true;
        }
        let packed = pack_rows(&rows, width, true);
        let enc = encode_ccitt_fax4(&packed, width, height, true);
        assert!(
            enc.len() < packed.len() / 4,
            "G4 should compress mostly-white image: {} vs {}",
            enc.len(),
            packed.len()
        );
        let dec = decode_ccitt_fax4(&enc, width, height, true).expect("fax4 ratio");
        assert_eq!(dec, packed);
    }

    #[test]
    fn test_fax3_compression_ratio() {
        let width = 256;
        let height = 64;
        let rows: Vec<Vec<bool>> = vec![vec![false; width]; height];
        let packed = pack_rows(&rows, width, true);
        let enc = encode_ccitt_fax3(&packed, width, height, true);
        assert!(
            enc.len() < packed.len(),
            "G3 should compress all-white image"
        );
        let dec = decode_ccitt_fax3(&enc, width, height, true).expect("fax3 ratio");
        assert_eq!(dec, packed);
    }

    #[test]
    fn test_white_is_zero_polarity() {
        let row0: &[bool] = &[true, true, false, false, true, true, false, false];
        let packed_wiz = pack_rows(&[row0.to_vec()], 8, true);
        let packed_biz = pack_rows(&[row0.to_vec()], 8, false);
        // The two packings are bit-inverses of each other.
        assert_eq!(packed_wiz[0], !packed_biz[0]);
        let enc = encode_ccitt_fax4(&packed_biz, 8, 1, false);
        let dec = decode_ccitt_fax4(&enc, 8, 1, false).expect("biz decode");
        assert_eq!(dec, packed_biz);
    }

    #[test]
    fn test_bit_writer_reader_roundtrip() {
        let mut w = BitWriter::new();
        let pattern = [(3u8, 0b101u16), (5, 0b11010), (1, 0b1), (12, 0x001)];
        for &(len, code) in &pattern {
            w.write_code(len, code);
        }
        assert_eq!(w.bit_count(), 3 + 5 + 1 + 12);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        for &(len, code) in &pattern {
            assert_eq!(r.peek(len), u32::from(code));
            r.skip(len);
        }
    }
}
