//! Variable Length Coding (VLC) tables for H.263.
//!
//! This module provides all the VLC tables used in H.263 encoding/decoding:
//! - TCOEF (Transform Coefficient) tables
//! - MVD (Motion Vector Difference) tables
//! - MCBPC (Macroblock type and Coded Block Pattern for Chrominance)
//! - CBPY (Coded Block Pattern for luminance)
//! - MODB (Macroblock mode in PB-frames)
//!
//! # References
//!
//! ITU-T Recommendation H.263 (1998) - Annex D: Variable length code tables

/// VLC entry containing code and bit length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VlcEntry {
    /// The VLC code value.
    pub code: u32,
    /// Number of bits in the code.
    pub bits: u8,
}

impl VlcEntry {
    /// Create a new VLC entry.
    #[must_use]
    pub const fn new(code: u32, bits: u8) -> Self {
        Self { code, bits }
    }
}

/// TCOEF (Transform Coefficient) entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcoefEntry {
    /// Run length of zeros.
    pub run: u8,
    /// Coefficient level.
    pub level: i16,
    /// Last coefficient flag.
    pub last: bool,
}

impl TcoefEntry {
    /// Create a new TCOEF entry.
    #[must_use]
    pub const fn new(run: u8, level: i16, last: bool) -> Self {
        Self { run, level, last }
    }
}

/// MCBPC (Macroblock type and CBPC) for I-frames.
///
/// Format: (mb_type, cbpc)
/// - mb_type: 0=Inter, 1=Inter+Q, 2=Inter4V, 3=Intra, 4=Intra+Q
/// - cbpc: Coded block pattern for chrominance (0-3)
pub const MCBPC_I: &[(u8, u8)] = &[
    (3, 0), // 1
    (3, 1), // 01
    (3, 2), // 001
    (3, 3), // 0001
    (4, 0), // 0000 1
    (4, 1), // 0000 01
    (4, 2), // 0000 001
    (4, 3), // 0000 0001
];

/// MCBPC for P-frames.
///
/// Format: (mb_type, cbpc)
/// - mb_type: 0=Inter, 1=Inter+Q, 2=Inter4V, 3=Intra, 4=Intra+Q
/// - cbpc: Coded block pattern for chrominance (0-3)
pub const MCBPC_P: &[(u8, u8)] = &[
    (0, 0), // 1
    (0, 1), // 0011
    (0, 2), // 0010
    (0, 3), // 0001 01
    (3, 0), // 0001 1
    (1, 0), // 0001 00
    (3, 1), // 0000 0011
    (1, 1), // 0000 0010
    (3, 2), // 0000 0001 1
    (1, 2), // 0000 0001 0
    (3, 3), // 0000 0000 11
    (1, 3), // 0000 0000 10
    (2, 0), // 0000 0111
    (4, 0), // 0000 0110
    (2, 1), // 0000 0101
    (4, 1), // 0000 0100
    (2, 2), // 0000 0001 1
    (4, 2), // 0000 0001 0
    (2, 3), // 0000 0000 01
    (4, 3), // 0000 0000 001
];

/// VLC codes for MCBPC in I-frames.
pub const MCBPC_I_VLC: &[VlcEntry] = &[
    VlcEntry::new(0b1, 1),        // Intra, CBPC=0
    VlcEntry::new(0b01, 2),       // Intra, CBPC=1
    VlcEntry::new(0b001, 3),      // Intra, CBPC=2
    VlcEntry::new(0b0001, 4),     // Intra, CBPC=3
    VlcEntry::new(0b00001, 5),    // Intra+Q, CBPC=0
    VlcEntry::new(0b000001, 6),   // Intra+Q, CBPC=1
    VlcEntry::new(0b0000001, 7),  // Intra+Q, CBPC=2
    VlcEntry::new(0b00000001, 8), // Intra+Q, CBPC=3
];

/// VLC codes for MCBPC in P-frames.
pub const MCBPC_P_VLC: &[VlcEntry] = &[
    VlcEntry::new(0b1, 1),            // Inter, CBPC=0
    VlcEntry::new(0b0011, 4),         // Inter, CBPC=1
    VlcEntry::new(0b0010, 4),         // Inter, CBPC=2
    VlcEntry::new(0b000101, 6),       // Inter, CBPC=3
    VlcEntry::new(0b00011, 5),        // Intra, CBPC=0
    VlcEntry::new(0b000100, 6),       // Inter+Q, CBPC=0
    VlcEntry::new(0b00000011, 8),     // Intra, CBPC=1
    VlcEntry::new(0b00000010, 8),     // Inter+Q, CBPC=1
    VlcEntry::new(0b000000011, 9),    // Intra, CBPC=2
    VlcEntry::new(0b000000010, 9),    // Inter+Q, CBPC=2
    VlcEntry::new(0b00000000011, 11), // Intra, CBPC=3
    VlcEntry::new(0b00000000010, 11), // Inter+Q, CBPC=3
    VlcEntry::new(0b00000111, 8),     // Inter4V, CBPC=0
    VlcEntry::new(0b00000110, 8),     // Intra+Q, CBPC=0
    VlcEntry::new(0b00000101, 8),     // Inter4V, CBPC=1
    VlcEntry::new(0b00000100, 8),     // Intra+Q, CBPC=1
    VlcEntry::new(0b000000011, 9),    // Inter4V, CBPC=2
    VlcEntry::new(0b000000010, 9),    // Intra+Q, CBPC=2
    VlcEntry::new(0b0000000001, 10),  // Inter4V, CBPC=3
    VlcEntry::new(0b00000000001, 11), // Intra+Q, CBPC=3
];

/// CBPY (Coded Block Pattern for Y) table.
///
/// Index is the pattern (0-15), value is the VLC code.
/// Pattern bits: [Y0, Y1, Y2, Y3] where 1 = coded, 0 = not coded.
pub const CBPY_VLC: &[VlcEntry] = &[
    VlcEntry::new(0b0011, 4),  // 0000 - none coded
    VlcEntry::new(0b00101, 5), // 0001 - Y3
    VlcEntry::new(0b00111, 5), // 0010 - Y2
    VlcEntry::new(0b01001, 5), // 0011 - Y2,Y3
    VlcEntry::new(0b01011, 5), // 0100 - Y1
    VlcEntry::new(0b01101, 5), // 0101 - Y1,Y3
    VlcEntry::new(0b01111, 5), // 0110 - Y1,Y2
    VlcEntry::new(0b00001, 5), // 0111 - Y1,Y2,Y3
    VlcEntry::new(0b10001, 5), // 1000 - Y0
    VlcEntry::new(0b10011, 5), // 1001 - Y0,Y3
    VlcEntry::new(0b10101, 5), // 1010 - Y0,Y2
    VlcEntry::new(0b10111, 5), // 1011 - Y0,Y2,Y3
    VlcEntry::new(0b11001, 5), // 1100 - Y0,Y1
    VlcEntry::new(0b11011, 5), // 1101 - Y0,Y1,Y3
    VlcEntry::new(0b11101, 5), // 1110 - Y0,Y1,Y2
    VlcEntry::new(0b0010, 4),  // 1111 - all coded
];

/// Inverted CBPY table for intra macroblocks.
///
/// For intra MBs, the CBPY pattern is inverted (1 = not coded, 0 = coded).
pub const CBPY_INTRA_VLC: &[VlcEntry] = &[
    VlcEntry::new(0b0010, 4),  // 1111 - all coded
    VlcEntry::new(0b11101, 5), // 1110
    VlcEntry::new(0b11011, 5), // 1101
    VlcEntry::new(0b11001, 5), // 1100
    VlcEntry::new(0b10111, 5), // 1011
    VlcEntry::new(0b10101, 5), // 1010
    VlcEntry::new(0b10011, 5), // 1001
    VlcEntry::new(0b10001, 5), // 1000
    VlcEntry::new(0b00001, 5), // 0111
    VlcEntry::new(0b01111, 5), // 0110
    VlcEntry::new(0b01101, 5), // 0101
    VlcEntry::new(0b01011, 5), // 0100
    VlcEntry::new(0b01001, 5), // 0011
    VlcEntry::new(0b00111, 5), // 0010
    VlcEntry::new(0b00101, 5), // 0001
    VlcEntry::new(0b0011, 4),  // 0000 - none coded
];

/// MVD (Motion Vector Difference) table.
///
/// VLC codes for motion vector differences.
/// Index represents the MVD value + 32 (range: -32 to +32).
pub const MVD_VLC: &[VlcEntry] = &[
    // -32 to -17
    VlcEntry::new(0b0000000001101, 13), // -32
    VlcEntry::new(0b0000000001111, 13), // -31
    VlcEntry::new(0b0000000010001, 13), // -30
    VlcEntry::new(0b0000000010011, 13), // -29
    VlcEntry::new(0b0000000010101, 13), // -28
    VlcEntry::new(0b0000000010111, 13), // -27
    VlcEntry::new(0b0000000011001, 13), // -26
    VlcEntry::new(0b0000000011011, 13), // -25
    VlcEntry::new(0b0000000011101, 13), // -24
    VlcEntry::new(0b0000000011111, 13), // -23
    VlcEntry::new(0b0000000100001, 13), // -22
    VlcEntry::new(0b0000000100011, 13), // -21
    VlcEntry::new(0b0000000100101, 13), // -20
    VlcEntry::new(0b0000000100111, 13), // -19
    VlcEntry::new(0b0000000101001, 13), // -18
    VlcEntry::new(0b0000000101011, 13), // -17
    // -16 to -9
    VlcEntry::new(0b00000001011, 11), // -16
    VlcEntry::new(0b00000001111, 11), // -15
    VlcEntry::new(0b00000010011, 11), // -14
    VlcEntry::new(0b00000010111, 11), // -13
    VlcEntry::new(0b00000011011, 11), // -12
    VlcEntry::new(0b00000011111, 11), // -11
    VlcEntry::new(0b00000100011, 11), // -10
    VlcEntry::new(0b00000100111, 11), // -9
    // -8 to -5
    VlcEntry::new(0b0000001011, 10), // -8
    VlcEntry::new(0b0000001111, 10), // -7
    VlcEntry::new(0b0000010011, 10), // -6
    VlcEntry::new(0b0000010111, 10), // -5
    // -4 to -3
    VlcEntry::new(0b000000011, 9), // -4
    VlcEntry::new(0b000001011, 9), // -3
    // -2 to -1
    VlcEntry::new(0b0000011, 7), // -2
    VlcEntry::new(0b000011, 6),  // -1
    // 0
    VlcEntry::new(0b1, 1), // 0
    // +1 to +2
    VlcEntry::new(0b000010, 6),  // +1
    VlcEntry::new(0b0000010, 7), // +2
    // +3 to +4
    VlcEntry::new(0b000001010, 9), // +3
    VlcEntry::new(0b000000010, 9), // +4
    // +5 to +8
    VlcEntry::new(0b0000010110, 10), // +5
    VlcEntry::new(0b0000010010, 10), // +6
    VlcEntry::new(0b0000001110, 10), // +7
    VlcEntry::new(0b0000001010, 10), // +8
    // +9 to +16
    VlcEntry::new(0b00000100110, 11), // +9
    VlcEntry::new(0b00000100010, 11), // +10
    VlcEntry::new(0b00000011110, 11), // +11
    VlcEntry::new(0b00000011010, 11), // +12
    VlcEntry::new(0b00000010110, 11), // +13
    VlcEntry::new(0b00000010010, 11), // +14
    VlcEntry::new(0b00000001110, 11), // +15
    VlcEntry::new(0b00000001010, 11), // +16
    // +17 to +32
    VlcEntry::new(0b0000000101010, 13), // +17
    VlcEntry::new(0b0000000101000, 13), // +18
    VlcEntry::new(0b0000000100110, 13), // +19
    VlcEntry::new(0b0000000100100, 13), // +20
    VlcEntry::new(0b0000000100010, 13), // +21
    VlcEntry::new(0b0000000100000, 13), // +22
    VlcEntry::new(0b0000000011110, 13), // +23
    VlcEntry::new(0b0000000011100, 13), // +24
    VlcEntry::new(0b0000000011010, 13), // +25
    VlcEntry::new(0b0000000011000, 13), // +26
    VlcEntry::new(0b0000000010110, 13), // +27
    VlcEntry::new(0b0000000010100, 13), // +28
    VlcEntry::new(0b0000000010010, 13), // +29
    VlcEntry::new(0b0000000010000, 13), // +30
    VlcEntry::new(0b0000000001110, 13), // +31
    VlcEntry::new(0b0000000001100, 13), // +32
];

/// TCOEF (Transform Coefficient) VLC table.
///
/// Format: (last, run, level) -> VLC code
/// This is a simplified table; full implementation would have escape codes.
pub const TCOEF_VLC: &[(TcoefEntry, VlcEntry)] = &[
    // last=0, run=0
    (TcoefEntry::new(0, 1, false), VlcEntry::new(0b10, 2)),
    (TcoefEntry::new(0, 2, false), VlcEntry::new(0b0101, 4)),
    (TcoefEntry::new(0, 3, false), VlcEntry::new(0b00101, 5)),
    (TcoefEntry::new(0, 4, false), VlcEntry::new(0b0000110, 7)),
    (TcoefEntry::new(0, 5, false), VlcEntry::new(0b00100110, 8)),
    (TcoefEntry::new(0, 6, false), VlcEntry::new(0b00100001, 8)),
    (TcoefEntry::new(0, 7, false), VlcEntry::new(0b00011110, 8)),
    (TcoefEntry::new(0, 8, false), VlcEntry::new(0b00011011, 8)),
    (TcoefEntry::new(0, 9, false), VlcEntry::new(0b00011000, 8)),
    (TcoefEntry::new(0, 10, false), VlcEntry::new(0b00010101, 8)),
    (TcoefEntry::new(0, 11, false), VlcEntry::new(0b00010010, 8)),
    (TcoefEntry::new(0, 12, false), VlcEntry::new(0b00001111, 8)),
    // last=0, run=1
    (TcoefEntry::new(1, 1, false), VlcEntry::new(0b011, 3)),
    (TcoefEntry::new(1, 2, false), VlcEntry::new(0b000110, 6)),
    (TcoefEntry::new(1, 3, false), VlcEntry::new(0b00100101, 8)),
    // last=0, run=2
    (TcoefEntry::new(2, 1, false), VlcEntry::new(0b0100, 4)),
    (TcoefEntry::new(2, 2, false), VlcEntry::new(0b00100100, 8)),
    // last=0, run=3
    (TcoefEntry::new(3, 1, false), VlcEntry::new(0b000101, 6)),
    // last=0, run=4
    (TcoefEntry::new(4, 1, false), VlcEntry::new(0b00100011, 8)),
    // last=0, run=5
    (TcoefEntry::new(5, 1, false), VlcEntry::new(0b00100010, 8)),
    // last=0, run=6+
    (TcoefEntry::new(6, 1, false), VlcEntry::new(0b00100000, 8)),
    (TcoefEntry::new(7, 1, false), VlcEntry::new(0b00011111, 8)),
    (TcoefEntry::new(8, 1, false), VlcEntry::new(0b00011101, 8)),
    (TcoefEntry::new(9, 1, false), VlcEntry::new(0b00011100, 8)),
    (TcoefEntry::new(10, 1, false), VlcEntry::new(0b00011010, 8)),
    (TcoefEntry::new(11, 1, false), VlcEntry::new(0b00011001, 8)),
    (TcoefEntry::new(12, 1, false), VlcEntry::new(0b00010111, 8)),
    (TcoefEntry::new(13, 1, false), VlcEntry::new(0b00010110, 8)),
    (TcoefEntry::new(14, 1, false), VlcEntry::new(0b00010100, 8)),
    (TcoefEntry::new(15, 1, false), VlcEntry::new(0b00010011, 8)),
    (TcoefEntry::new(16, 1, false), VlcEntry::new(0b00010001, 8)),
    (TcoefEntry::new(17, 1, false), VlcEntry::new(0b00010000, 8)),
    (TcoefEntry::new(18, 1, false), VlcEntry::new(0b00001110, 8)),
    (TcoefEntry::new(19, 1, false), VlcEntry::new(0b00001101, 8)),
    (TcoefEntry::new(20, 1, false), VlcEntry::new(0b00001100, 8)),
    (TcoefEntry::new(21, 1, false), VlcEntry::new(0b00001011, 8)),
    (TcoefEntry::new(22, 1, false), VlcEntry::new(0b00001010, 8)),
    (TcoefEntry::new(23, 1, false), VlcEntry::new(0b00001001, 8)),
    (TcoefEntry::new(24, 1, false), VlcEntry::new(0b00001000, 8)),
    (TcoefEntry::new(25, 1, false), VlcEntry::new(0b00000111, 8)),
    (TcoefEntry::new(26, 1, false), VlcEntry::new(0b00000110, 8)),
    // last=1 entries
    (TcoefEntry::new(0, 1, true), VlcEntry::new(0b0110, 4)),
    (TcoefEntry::new(0, 2, true), VlcEntry::new(0b00111, 5)),
    (TcoefEntry::new(0, 3, true), VlcEntry::new(0b000111, 6)),
    (TcoefEntry::new(0, 4, true), VlcEntry::new(0b00011010, 8)),
    (TcoefEntry::new(0, 5, true), VlcEntry::new(0b00011001, 8)),
    (TcoefEntry::new(0, 6, true), VlcEntry::new(0b00010111, 8)),
    (TcoefEntry::new(1, 1, true), VlcEntry::new(0b00110, 5)),
    (TcoefEntry::new(1, 2, true), VlcEntry::new(0b00010110, 8)),
    (TcoefEntry::new(2, 1, true), VlcEntry::new(0b000100, 6)),
    (TcoefEntry::new(3, 1, true), VlcEntry::new(0b00010101, 8)),
];

/// Intra DC VLC table for luminance (Y).
pub const INTRA_DC_LUMA_VLC: &[VlcEntry] = &[
    VlcEntry::new(0b00, 2),        // 0
    VlcEntry::new(0b010, 3),       // 1
    VlcEntry::new(0b011, 3),       // 2
    VlcEntry::new(0b100, 3),       // 3
    VlcEntry::new(0b101, 3),       // 4
    VlcEntry::new(0b110, 3),       // 5
    VlcEntry::new(0b1110, 4),      // 6
    VlcEntry::new(0b11110, 5),     // 7
    VlcEntry::new(0b111110, 6),    // 8
    VlcEntry::new(0b1111110, 7),   // 9
    VlcEntry::new(0b11111110, 8),  // 10
    VlcEntry::new(0b111111110, 9), // 11
];

/// Intra DC VLC table for chrominance (Cb/Cr).
pub const INTRA_DC_CHROMA_VLC: &[VlcEntry] = &[
    VlcEntry::new(0b00, 2),           // 0
    VlcEntry::new(0b01, 2),           // 1
    VlcEntry::new(0b10, 2),           // 2
    VlcEntry::new(0b110, 3),          // 3
    VlcEntry::new(0b1110, 4),         // 4
    VlcEntry::new(0b11110, 5),        // 5
    VlcEntry::new(0b111110, 6),       // 6
    VlcEntry::new(0b1111110, 7),      // 7
    VlcEntry::new(0b11111110, 8),     // 8
    VlcEntry::new(0b111111110, 9),    // 9
    VlcEntry::new(0b1111111110, 10),  // 10
    VlcEntry::new(0b11111111110, 11), // 11
];

/// MODB (Macroblock mode in PB-frames) VLC table.
pub const MODB_VLC: &[VlcEntry] = &[
    VlcEntry::new(0b0, 1),  // MODB=0
    VlcEntry::new(0b10, 2), // MODB=1
    VlcEntry::new(0b11, 2), // MODB=2
];

/// Decode MVD (Motion Vector Difference) from VLC code.
///
/// # Arguments
///
/// * `code` - The VLC code value
/// * `bits` - Number of bits in the code
///
/// # Returns
///
/// The decoded MVD value, or None if invalid code.
#[must_use]
pub fn decode_mvd(code: u32, bits: u8) -> Option<i32> {
    for (idx, entry) in MVD_VLC.iter().enumerate() {
        if entry.code == code && entry.bits == bits {
            return Some((idx as i32) - 32);
        }
    }
    None
}

/// Encode MVD (Motion Vector Difference) to VLC code.
///
/// # Arguments
///
/// * `mvd` - Motion vector difference value (-32 to +32)
///
/// # Returns
///
/// The VLC entry for this MVD, or None if out of range.
#[must_use]
pub fn encode_mvd(mvd: i32) -> Option<VlcEntry> {
    if !(-32..=32).contains(&mvd) {
        return None;
    }
    let idx = (mvd + 32) as usize;
    Some(MVD_VLC[idx])
}

/// Decode CBPY (Coded Block Pattern for Y) from VLC code.
///
/// # Arguments
///
/// * `code` - The VLC code value
/// * `bits` - Number of bits in the code
/// * `intra` - True if intra macroblock
///
/// # Returns
///
/// The CBPY pattern (0-15), or None if invalid code.
#[must_use]
pub fn decode_cbpy(code: u32, bits: u8, intra: bool) -> Option<u8> {
    let table = if intra { CBPY_INTRA_VLC } else { CBPY_VLC };

    for (pattern, entry) in table.iter().enumerate() {
        if entry.code == code && entry.bits == bits {
            return Some(pattern as u8);
        }
    }
    None
}

/// Encode CBPY (Coded Block Pattern for Y) to VLC code.
///
/// # Arguments
///
/// * `pattern` - The CBPY pattern (0-15)
/// * `intra` - True if intra macroblock
///
/// # Returns
///
/// The VLC entry for this pattern, or None if out of range.
#[must_use]
pub fn encode_cbpy(pattern: u8, intra: bool) -> Option<VlcEntry> {
    if pattern > 15 {
        return None;
    }
    let table = if intra { CBPY_INTRA_VLC } else { CBPY_VLC };
    Some(table[pattern as usize])
}

/// Lookup table for fast TCOEF decoding.
///
/// This would be implemented as a hash table or search tree in practice.
pub fn find_tcoef_entry(code: u32, bits: u8) -> Option<TcoefEntry> {
    for (entry, vlc) in TCOEF_VLC {
        if vlc.code == code && vlc.bits == bits {
            return Some(*entry);
        }
    }
    None
}

/// Find VLC code for a TCOEF entry.
///
/// # Arguments
///
/// * `entry` - The TCOEF entry (run, level, last)
///
/// # Returns
///
/// The VLC code, or None if not found.
pub fn find_tcoef_vlc(entry: &TcoefEntry) -> Option<VlcEntry> {
    for (tcoef, vlc) in TCOEF_VLC {
        if tcoef.run == entry.run && tcoef.level == entry.level && tcoef.last == entry.last {
            return Some(*vlc);
        }
    }
    None
}

/// Decode MCBPC for I-frames.
///
/// # Arguments
///
/// * `code` - The VLC code value
/// * `bits` - Number of bits in the code
///
/// # Returns
///
/// (mb_type, cbpc), or None if invalid code.
#[must_use]
pub fn decode_mcbpc_i(code: u32, bits: u8) -> Option<(u8, u8)> {
    for (idx, entry) in MCBPC_I_VLC.iter().enumerate() {
        if entry.code == code && entry.bits == bits {
            if idx < MCBPC_I.len() {
                return Some(MCBPC_I[idx]);
            }
        }
    }
    None
}

/// Decode MCBPC for P-frames.
///
/// # Arguments
///
/// * `code` - The VLC code value
/// * `bits` - Number of bits in the code
///
/// # Returns
///
/// (mb_type, cbpc), or None if invalid code.
#[must_use]
pub fn decode_mcbpc_p(code: u32, bits: u8) -> Option<(u8, u8)> {
    for (idx, entry) in MCBPC_P_VLC.iter().enumerate() {
        if entry.code == code && entry.bits == bits {
            if idx < MCBPC_P.len() {
                return Some(MCBPC_P[idx]);
            }
        }
    }
    None
}

/// Encode MCBPC for I-frames.
///
/// # Arguments
///
/// * `mb_type` - Macroblock type
/// * `cbpc` - Coded block pattern for chrominance
///
/// # Returns
///
/// VLC entry, or None if invalid.
#[must_use]
pub fn encode_mcbpc_i(mb_type: u8, cbpc: u8) -> Option<VlcEntry> {
    for (idx, &(mbt, cbp)) in MCBPC_I.iter().enumerate() {
        if mbt == mb_type && cbp == cbpc && idx < MCBPC_I_VLC.len() {
            return Some(MCBPC_I_VLC[idx]);
        }
    }
    None
}

/// Encode MCBPC for P-frames.
///
/// # Arguments
///
/// * `mb_type` - Macroblock type
/// * `cbpc` - Coded block pattern for chrominance
///
/// # Returns
///
/// VLC entry, or None if invalid.
#[must_use]
pub fn encode_mcbpc_p(mb_type: u8, cbpc: u8) -> Option<VlcEntry> {
    for (idx, &(mbt, cbp)) in MCBPC_P.iter().enumerate() {
        if mbt == mb_type && cbp == cbpc && idx < MCBPC_P_VLC.len() {
            return Some(MCBPC_P_VLC[idx]);
        }
    }
    None
}
