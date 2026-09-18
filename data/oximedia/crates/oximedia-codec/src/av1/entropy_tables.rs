//! AV1 entropy coding default CDF tables.
//!
//! This module contains the default probability tables (CDFs) used for
//! entropy coding in AV1. These tables are used as initial values and
//! are updated during decoding based on observed symbol frequencies.
//!
//! # CDF Format
//!
//! CDFs are stored as arrays of 16-bit unsigned integers. The last element
//! is reserved for the symbol count used in CDF update. The actual
//! probabilities are in 15-bit precision (0-32767).
//!
//! # Table Categories
//!
//! - **Partition CDFs** - Block partitioning decisions
//! - **Intra Mode CDFs** - Intra prediction mode selection
//! - **TX Size CDFs** - Transform size selection
//! - **TX Type CDFs** - Transform type selection
//! - **Coefficient CDFs** - Coefficient level and sign coding
//! - **MV Component CDFs** - Motion vector component coding
//!
//! # Reference
//!
//! See AV1 Specification Section 9 for probability model initialization
//! and update procedures.

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_range_loop)]

// =============================================================================
// CDF Precision Constants
// =============================================================================

/// CDF precision bits.
pub const CDF_PROB_BITS: u8 = 15;

/// Maximum CDF probability value.
pub const CDF_PROB_TOP: u16 = 1 << CDF_PROB_BITS;

/// Initial symbol count for CDF adaptation.
pub const CDF_INIT_COUNT: u16 = 0;

/// Maximum symbol count for CDF adaptation rate.
pub const CDF_MAX_COUNT: u16 = 32;

// =============================================================================
// Partition CDFs
// =============================================================================

/// Number of partition contexts.
pub const PARTITION_CONTEXTS: usize = 4;

/// Number of partition types.
pub const PARTITION_TYPES: usize = 10;

/// Default CDF for partition (context 0, small blocks).
pub const DEFAULT_PARTITION_CDF_0: [u16; 11] = [
    15588, 17570, 19323, 21084, 22472, 24311, 25744, 27999, 29223, 32768, 0,
];

/// Default CDF for partition (context 1, medium blocks).
pub const DEFAULT_PARTITION_CDF_1: [u16; 11] = [
    12064, 14616, 17239, 19824, 21631, 24068, 25919, 28400, 29760, 32768, 0,
];

/// Default CDF for partition (context 2, large blocks).
pub const DEFAULT_PARTITION_CDF_2: [u16; 11] = [
    9216, 12096, 15424, 18432, 20672, 23424, 25664, 28544, 30080, 32768, 0,
];

/// Default CDF for partition (context 3, very large blocks).
pub const DEFAULT_PARTITION_CDF_3: [u16; 11] = [
    6144, 9472, 13312, 16896, 19584, 22912, 25600, 28800, 30464, 32768, 0,
];

/// All default partition CDFs.
pub const DEFAULT_PARTITION_CDFS: [[u16; 11]; PARTITION_CONTEXTS] = [
    DEFAULT_PARTITION_CDF_0,
    DEFAULT_PARTITION_CDF_1,
    DEFAULT_PARTITION_CDF_2,
    DEFAULT_PARTITION_CDF_3,
];

// =============================================================================
// Intra Mode CDFs
// =============================================================================

/// Number of intra modes.
pub const INTRA_MODES: usize = 13;

/// Number of intra mode contexts for Y.
pub const INTRA_Y_MODE_CONTEXTS: usize = 4;

/// Default CDF for Y intra mode (context 0).
pub const DEFAULT_Y_MODE_CDF_0: [u16; 14] = [
    15588, 17570, 18800, 20000, 21500, 23000, 24500, 26000, 27500, 29000, 30500, 31500, 32768, 0,
];

/// Default CDF for Y intra mode (context 1).
pub const DEFAULT_Y_MODE_CDF_1: [u16; 14] = [
    12064, 14616, 16500, 18500, 20500, 22500, 24500, 26500, 28000, 29500, 30750, 31750, 32768, 0,
];

/// Default CDF for Y intra mode (context 2).
pub const DEFAULT_Y_MODE_CDF_2: [u16; 14] = [
    9216, 12096, 14500, 17000, 19500, 22000, 24500, 26500, 28500, 30000, 31000, 32000, 32768, 0,
];

/// Default CDF for Y intra mode (context 3).
pub const DEFAULT_Y_MODE_CDF_3: [u16; 14] = [
    6144, 9472, 12500, 15500, 18500, 21500, 24500, 27000, 29000, 30500, 31250, 32000, 32768, 0,
];

/// All default Y mode CDFs.
pub const DEFAULT_Y_MODE_CDFS: [[u16; 14]; INTRA_Y_MODE_CONTEXTS] = [
    DEFAULT_Y_MODE_CDF_0,
    DEFAULT_Y_MODE_CDF_1,
    DEFAULT_Y_MODE_CDF_2,
    DEFAULT_Y_MODE_CDF_3,
];

/// Number of UV intra mode contexts.
pub const INTRA_UV_MODE_CONTEXTS: usize = 13;

/// Default CDF for UV intra mode (for CFL disabled).
pub const DEFAULT_UV_MODE_CDF_NO_CFL: [u16; 14] = [
    22528, 24320, 25344, 26368, 27136, 28160, 28928, 29696, 30464, 31104, 31616, 32128, 32768, 0,
];

/// Default CDF for UV intra mode (for CFL enabled).
pub const DEFAULT_UV_MODE_CDF_CFL: [u16; 15] = [
    18432, 20480, 22016, 23296, 24576, 25856, 27136, 28160, 29184, 30080, 30848, 31488, 32000,
    32768, 0,
];

// =============================================================================
// Transform Size CDFs
// =============================================================================

/// Number of TX size contexts.
pub const TX_SIZE_CONTEXTS: usize = 3;

/// Number of max TX size categories.
pub const MAX_TX_CATS: usize = 4;

/// Default CDF for TX size (max 8x8).
pub const DEFAULT_TX_SIZE_CDF_8X8: [u16; 3] = [16384, 32768, 0];

/// Default CDF for TX size (max 16x16).
pub const DEFAULT_TX_SIZE_CDF_16X16: [u16; 4] = [10923, 21845, 32768, 0];

/// Default CDF for TX size (max 32x32).
pub const DEFAULT_TX_SIZE_CDF_32X32: [u16; 5] = [8192, 16384, 24576, 32768, 0];

/// Default CDF for TX size (max 64x64).
pub const DEFAULT_TX_SIZE_CDF_64X64: [u16; 6] = [6554, 13107, 19661, 26214, 32768, 0];

// =============================================================================
// Transform Type CDFs
// =============================================================================

/// Number of TX type contexts per set.
pub const TX_TYPE_CONTEXTS: usize = 7;

/// Number of transform types for intra.
pub const INTRA_TX_TYPES: usize = 7;

/// Number of transform types for inter.
pub const INTER_TX_TYPES: usize = 16;

/// Default CDF for intra TX type (TX_4X4).
pub const DEFAULT_INTRA_TX_TYPE_4X4: [[u16; 8]; TX_TYPE_CONTEXTS] = [
    [5461, 10923, 16384, 21845, 24576, 27307, 30037, 32768],
    [4681, 9362, 14043, 18725, 22118, 25512, 28905, 32768],
    [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768],
    [3641, 7282, 10923, 14564, 18893, 23222, 27551, 32768],
    [3277, 6554, 9830, 13107, 17476, 21845, 26214, 32768],
    [2979, 5958, 8937, 11916, 16213, 20511, 25228, 32768],
    [2731, 5461, 8192, 10923, 15019, 19114, 24064, 32768],
];

/// Default CDF for intra TX type (TX_8X8).
pub const DEFAULT_INTRA_TX_TYPE_8X8: [[u16; 8]; TX_TYPE_CONTEXTS] = [
    [6144, 12288, 18432, 24576, 26624, 28672, 30720, 32768],
    [5461, 10923, 16384, 21845, 24576, 27307, 30037, 32768],
    [4915, 9830, 14745, 19660, 22938, 26214, 29491, 32768],
    [4455, 8909, 13364, 17818, 21399, 24980, 28561, 32768],
    [4096, 8192, 12288, 16384, 20070, 23756, 27852, 32768],
    [3780, 7559, 11339, 15119, 18897, 22675, 27200, 32768],
    [3495, 6991, 10486, 13981, 17827, 21673, 26600, 32768],
];

/// Default CDF for inter TX type.
pub const DEFAULT_INTER_TX_TYPE: [[u16; 17]; TX_TYPE_CONTEXTS] = [
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
];

// =============================================================================
// Coefficient CDFs
// =============================================================================

/// Number of EOB multi contexts per plane type.
///
/// Per AV1 Annex F §9.5, EOB symbol coding is parameterised by the transform
/// area class. The seven classes correspond to areas 16, 32, 64, 128, 256,
/// 512, and 1024+ — matching the seven distinct `eob_multi*_cdf` tables in
/// the reference decoder (`EobMultiSize16Cdf` through `EobMultiSize1024Cdf`).
pub const EOB_MULTI_CONTEXTS: usize = 7;

/// Number of transform sizes in AV1 (matches `TxSize` enum, indices 0..=18).
pub const TX_SIZE_COUNT: usize = 19;

/// Number of plane types (luma vs. one of two chroma planes).
pub const EOB_PLANE_COUNT: usize = 3;

/// Number of `(tx_size, plane)` contexts addressed by the caller.
///
/// The decoder/encoder packs the EOB context as
/// `ctx = tx_size_idx * EOB_PLANE_COUNT + plane`, with `tx_size_idx` in
/// `0..TX_SIZE_COUNT` and `plane` in `0..EOB_PLANE_COUNT`. The product is
/// the number of distinct CDF slots maintained per frame.
pub const EOB_MULTI_TOTAL_CONTEXTS: usize = TX_SIZE_COUNT * EOB_PLANE_COUNT;

/// Default CDF for EOB multi (2 symbols).
pub const DEFAULT_EOB_MULTI_2: [u16; 3] = [16384, 32768, 0];

/// Default CDF for EOB multi (4 symbols).
pub const DEFAULT_EOB_MULTI_4: [u16; 5] = [8192, 16384, 24576, 32768, 0];

/// Default CDF for EOB multi (8 symbols).
pub const DEFAULT_EOB_MULTI_8: [u16; 9] = [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 0];

/// Default CDF for EOB multi (16 symbols).
pub const DEFAULT_EOB_MULTI_16: [u16; 17] = [
    2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672,
    30720, 32768, 0,
];

/// Map a `TxSize` integer index to the appropriate default EOB-multi CDF.
///
/// The class is chosen by transform area, matching the maximum value of
/// [`crate::av1::coefficients::EobPt::from_eob`] for that area:
///
/// | Area   | TxSizes                       | max `EobPt` | CDF picked |
/// |--------|-------------------------------|-------------|------------|
/// | 16     | 4×4                           | 5           | `_8` (9)   |
/// | 32     | 4×8, 8×4                      | 6           | `_8` (9)   |
/// | 64     | 8×8, 4×16, 16×4               | 7           | `_8` (9)   |
/// | 128    | 8×16, 16×8                    | 8           | `_16` (17) |
/// | 256    | 16×16, 8×32, 32×8             | 9           | `_16` (17) |
/// | 512    | 16×32, 32×16                  | 10          | `_16` (17) |
/// | ≥ 1024 | 32×32, 32×64, 64×32, 64×64, … | 11          | `_16` (17) |
///
/// `DEFAULT_EOB_MULTI_2` and `DEFAULT_EOB_MULTI_4` are retained for
/// reference (they appear in the public spec text) but are not selected
/// because no transform area yields a max EOB point ≤ 3 in the present
/// decoder.
#[must_use]
fn default_eob_multi_for_tx_size_idx(tx_size_idx: usize) -> Vec<u16> {
    // Areas of TxSize variants in declaration order. Out-of-range
    // indices fall through to the safest (largest) bucket.
    const TX_SIZE_AREAS: [u32; TX_SIZE_COUNT] = [
        16,   // Tx4x4
        64,   // Tx8x8
        256,  // Tx16x16
        1024, // Tx32x32
        4096, // Tx64x64
        32,   // Tx4x8
        32,   // Tx8x4
        128,  // Tx8x16
        128,  // Tx16x8
        512,  // Tx16x32
        512,  // Tx32x16
        2048, // Tx32x64
        2048, // Tx64x32
        64,   // Tx4x16
        64,   // Tx16x4
        256,  // Tx8x32
        256,  // Tx32x8
        1024, // Tx16x64
        1024, // Tx64x16
    ];

    let area = TX_SIZE_AREAS.get(tx_size_idx).copied().unwrap_or(4096);

    if area <= 64 {
        DEFAULT_EOB_MULTI_8.to_vec()
    } else {
        DEFAULT_EOB_MULTI_16.to_vec()
    }
}

/// Number of coefficient base contexts.
pub const COEFF_BASE_CTX_COUNT: usize = 42;

/// Default CDF for coefficient base (4 levels: 0, 1, 2, >2).
pub const DEFAULT_COEFF_BASE_CDF: [u16; 5] = [8192, 16384, 24576, 32768, 0];

/// Default CDF for coefficient base EOB.
pub const DEFAULT_COEFF_BASE_EOB_CDF: [u16; 4] = [10923, 21845, 32768, 0];

/// Number of DC sign contexts.
pub const DC_SIGN_CTX_COUNT: usize = 3;

/// Default CDF for DC sign.
pub const DEFAULT_DC_SIGN_CDF: [u16; 3] = [16384, 32768, 0];

/// Number of coefficient base range contexts.
pub const COEFF_BR_CTX_COUNT: usize = 21;

/// Default CDF for coefficient base range.
pub const DEFAULT_COEFF_BR_CDF: [u16; 4] = [10923, 21845, 32768, 0];

// =============================================================================
// Motion Vector CDFs
// =============================================================================

/// Number of MV joint types.
pub const MV_JOINTS: usize = 4;

/// Default CDF for MV joint.
pub const DEFAULT_MV_JOINT_CDF: [u16; 5] = [4096, 11264, 19712, 32768, 0];

/// Number of MV classes.
pub const MV_CLASSES: usize = 11;

/// Default CDF for MV class.
pub const DEFAULT_MV_CLASS_CDF: [u16; 12] = [
    28672, 30976, 31744, 32128, 32320, 32448, 32544, 32608, 32672, 32720, 32768, 0,
];

/// Default CDF for MV class 0 bit.
pub const DEFAULT_MV_CLASS0_BIT_CDF: [u16; 3] = [16384, 32768, 0];

/// Number of MV class 0 fractional values.
pub const MV_CLASS0_FP: usize = 4;

/// Default CDF for MV class 0 fractional.
pub const DEFAULT_MV_CLASS0_FP_CDF: [[u16; 5]; 2] = [
    [8192, 16384, 24576, 32768, 0],
    [8192, 16384, 24576, 32768, 0],
];

/// Number of MV fractional values.
pub const MV_FP: usize = 4;

/// Default CDF for MV fractional.
pub const DEFAULT_MV_FP_CDF: [u16; 5] = [8192, 16384, 24576, 32768, 0];

/// Default CDF for MV class 0 high precision.
pub const DEFAULT_MV_CLASS0_HP_CDF: [u16; 3] = [16384, 32768, 0];

/// Default CDF for MV high precision.
pub const DEFAULT_MV_HP_CDF: [u16; 3] = [16384, 32768, 0];

/// Default CDF for MV sign.
pub const DEFAULT_MV_SIGN_CDF: [u16; 3] = [16384, 32768, 0];

/// Number of MV bits for class > 0.
pub const MV_OFFSET_BITS: usize = 10;

/// Default CDF for MV bits.
pub const DEFAULT_MV_BITS_CDF: [[u16; 3]; MV_OFFSET_BITS] = [
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
];

// =============================================================================
// Skip CDFs
// =============================================================================

/// Number of skip contexts.
pub const SKIP_CONTEXTS: usize = 3;

/// Default CDF for skip.
pub const DEFAULT_SKIP_CDF: [[u16; 3]; SKIP_CONTEXTS] =
    [[24576, 32768, 0], [16384, 32768, 0], [8192, 32768, 0]];

// =============================================================================
// Segment CDFs
// =============================================================================

/// Maximum number of segments.
pub const MAX_SEGMENTS: usize = 8;

/// Default CDF for segment ID (tree).
pub const DEFAULT_SEGMENT_TREE_CDF: [u16; 9] =
    [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 0];

/// Default CDF for segment ID prediction.
pub const DEFAULT_SEGMENT_PRED_CDF: [[u16; 3]; 3] =
    [[16384, 32768, 0], [16384, 32768, 0], [16384, 32768, 0]];

// =============================================================================
// Reference Frame CDFs
// =============================================================================

/// Number of reference frame contexts.
pub const REF_CONTEXTS: usize = 3;

/// Number of reference frame types for single ref.
pub const SINGLE_REF_TYPES: usize = 7;

/// Default CDF for single reference frame.
pub const DEFAULT_SINGLE_REF_CDF: [[[u16; 3]; SINGLE_REF_TYPES]; REF_CONTEXTS] = [
    [
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
    ],
    [
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
    ],
    [
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
        [16384, 32768, 0],
    ],
];

// =============================================================================
// Inter Mode CDFs
// =============================================================================

/// Number of inter mode contexts.
pub const INTER_MODE_CONTEXTS: usize = 8;

/// Number of inter modes.
pub const INTER_MODES: usize = 4;

/// Default CDF for inter mode.
pub const DEFAULT_INTER_MODE_CDF: [[u16; 5]; INTER_MODE_CONTEXTS] = [
    [2048, 10240, 17664, 32768, 0],
    [4096, 12288, 20480, 32768, 0],
    [6144, 14336, 22528, 32768, 0],
    [8192, 16384, 24576, 32768, 0],
    [10240, 18432, 26624, 32768, 0],
    [12288, 20480, 28672, 32768, 0],
    [14336, 22528, 29696, 32768, 0],
    [16384, 24576, 30720, 32768, 0],
];

// =============================================================================
// Compound Mode CDFs
// =============================================================================

/// Number of compound mode contexts.
pub const COMPOUND_MODE_CONTEXTS: usize = 8;

/// Number of compound modes.
pub const COMPOUND_MODES: usize = 8;

/// Default CDF for compound mode.
pub const DEFAULT_COMPOUND_MODE_CDF: [[u16; 9]; COMPOUND_MODE_CONTEXTS] = [
    [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 0],
    [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 0],
    [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 0],
    [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 0],
    [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 0],
    [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 0],
    [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 0],
    [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 0],
];

// =============================================================================
// Filter CDFs
// =============================================================================

/// Number of interpolation filter types.
pub const INTERP_FILTERS: usize = 4;

/// Number of interpolation filter contexts.
pub const INTERP_FILTER_CONTEXTS: usize = 16;

/// Default CDF for interpolation filter.
pub const DEFAULT_INTERP_FILTER_CDF: [[u16; 5]; INTERP_FILTER_CONTEXTS] = [
    [6144, 12288, 18432, 32768, 0],
    [6144, 12288, 18432, 32768, 0],
    [6144, 12288, 18432, 32768, 0],
    [6144, 12288, 18432, 32768, 0],
    [8192, 16384, 24576, 32768, 0],
    [8192, 16384, 24576, 32768, 0],
    [8192, 16384, 24576, 32768, 0],
    [8192, 16384, 24576, 32768, 0],
    [10240, 18432, 26624, 32768, 0],
    [10240, 18432, 26624, 32768, 0],
    [10240, 18432, 26624, 32768, 0],
    [10240, 18432, 26624, 32768, 0],
    [12288, 20480, 28672, 32768, 0],
    [12288, 20480, 28672, 32768, 0],
    [12288, 20480, 28672, 32768, 0],
    [12288, 20480, 28672, 32768, 0],
];

// =============================================================================
// CDF Helper Functions
// =============================================================================

/// Create a uniform CDF for n symbols.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn create_uniform_cdf(n: usize) -> Vec<u16> {
    let mut cdf = Vec::with_capacity(n + 1);
    for i in 1..=n {
        cdf.push(((i * CDF_PROB_TOP as usize) / n) as u16);
    }
    cdf.push(CDF_INIT_COUNT); // Symbol count
    cdf
}

/// Update CDF after observing a symbol.
#[allow(clippy::cast_possible_truncation)]
pub fn update_cdf(cdf: &mut [u16], symbol: usize) {
    let n = cdf.len() - 1; // Last element is count
    if n == 0 {
        return;
    }

    // Get current count and compute rate
    let count = u32::from(cdf[n]);
    let rate = 3 + (count >> 4);
    let rate = rate.min(32);

    // Update CDF values
    for i in 0..n {
        if i < symbol {
            // Decrease probability
            let diff = cdf[i] >> rate;
            cdf[i] = cdf[i].saturating_sub(diff);
        } else {
            // Increase probability
            let diff = CDF_PROB_TOP.saturating_sub(cdf[i]) >> rate;
            cdf[i] = cdf[i].saturating_add(diff);
        }
    }

    // Increment count
    if count < u32::from(CDF_MAX_COUNT) {
        cdf[n] += 1;
    }
}

/// Reset CDF to uniform distribution.
#[allow(clippy::cast_possible_truncation)]
pub fn reset_cdf_uniform(cdf: &mut [u16]) {
    let n = cdf.len() - 1;
    if n == 0 {
        return;
    }

    for i in 0..n {
        cdf[i] = (((i + 1) * CDF_PROB_TOP as usize) / n) as u16;
    }
    cdf[n] = CDF_INIT_COUNT;
}

/// Copy CDF from source to destination.
pub fn copy_cdf(dst: &mut [u16], src: &[u16]) {
    let len = dst.len().min(src.len());
    dst[..len].copy_from_slice(&src[..len]);
}

/// Check if a CDF is valid (monotonically increasing, ends at CDF_PROB_TOP).
#[must_use]
pub fn is_valid_cdf(cdf: &[u16]) -> bool {
    if cdf.len() < 2 {
        return false;
    }

    let n = cdf.len() - 1;

    // Check monotonicity
    for i in 1..n {
        if cdf[i] < cdf[i - 1] {
            return false;
        }
    }

    // Last probability should be CDF_PROB_TOP
    cdf[n - 1] == CDF_PROB_TOP
}

// =============================================================================
// CDF Context Management
// =============================================================================

/// Container for all CDF tables used in decoding.
#[derive(Clone, Debug)]
pub struct CdfContext {
    /// Partition CDFs.
    pub partition: [[u16; 11]; PARTITION_CONTEXTS],
    /// Y intra mode CDFs.
    pub y_mode: [[u16; 14]; INTRA_Y_MODE_CONTEXTS],
    /// Skip CDFs.
    pub skip: [[u16; 3]; SKIP_CONTEXTS],
    /// MV joint CDF.
    pub mv_joint: [u16; 5],
    /// MV sign CDFs (for each component).
    pub mv_sign: [[u16; 3]; 2],
    /// MV class CDFs.
    pub mv_class: [[u16; 12]; 2],
    /// DC sign CDFs.
    pub dc_sign: [[u16; 3]; DC_SIGN_CTX_COUNT],
    /// Coefficient base CDFs.
    pub coeff_base: Vec<[u16; 5]>,
    /// Coefficient base range CDFs.
    pub coeff_br: Vec<[u16; 4]>,
    /// EOB multi-symbol CDFs.
    ///
    /// Indexed by `ctx = tx_size_idx * EOB_PLANE_COUNT + plane` (see
    /// `EOB_MULTI_TOTAL_CONTEXTS`). Each inner `Vec<u16>` is a CDF whose
    /// length is selected per transform area — `_8` (9 entries) for small
    /// blocks (area ≤ 64) and `_16` (17 entries) for larger blocks.
    pub eob_multi: Vec<Vec<u16>>,
}

impl CdfContext {
    /// Create a new CDF context with default values.
    #[must_use]
    pub fn new() -> Self {
        let eob_multi = (0..EOB_MULTI_TOTAL_CONTEXTS)
            .map(|ctx| default_eob_multi_for_tx_size_idx(ctx / EOB_PLANE_COUNT))
            .collect();

        Self {
            partition: DEFAULT_PARTITION_CDFS,
            y_mode: DEFAULT_Y_MODE_CDFS,
            skip: DEFAULT_SKIP_CDF,
            mv_joint: DEFAULT_MV_JOINT_CDF,
            mv_sign: [DEFAULT_MV_SIGN_CDF, DEFAULT_MV_SIGN_CDF],
            mv_class: [DEFAULT_MV_CLASS_CDF, DEFAULT_MV_CLASS_CDF],
            dc_sign: [DEFAULT_DC_SIGN_CDF; DC_SIGN_CTX_COUNT],
            coeff_base: vec![DEFAULT_COEFF_BASE_CDF; COEFF_BASE_CTX_COUNT],
            coeff_br: vec![DEFAULT_COEFF_BR_CDF; COEFF_BR_CTX_COUNT],
            eob_multi,
        }
    }

    /// Reset all CDFs to default values.
    pub fn reset(&mut self) {
        self.partition = DEFAULT_PARTITION_CDFS;
        self.y_mode = DEFAULT_Y_MODE_CDFS;
        self.skip = DEFAULT_SKIP_CDF;
        self.mv_joint = DEFAULT_MV_JOINT_CDF;
        self.mv_sign = [DEFAULT_MV_SIGN_CDF, DEFAULT_MV_SIGN_CDF];
        self.mv_class = [DEFAULT_MV_CLASS_CDF, DEFAULT_MV_CLASS_CDF];
        self.dc_sign = [DEFAULT_DC_SIGN_CDF; DC_SIGN_CTX_COUNT];

        for cdf in &mut self.coeff_base {
            *cdf = DEFAULT_COEFF_BASE_CDF;
        }

        for cdf in &mut self.coeff_br {
            *cdf = DEFAULT_COEFF_BR_CDF;
        }

        // Rebuild EOB multi CDFs so each ctx is reset to the proper default
        // for its `(tx_size, plane)` pair. We resize first to guarantee
        // length invariants even if a caller previously shrank the Vec.
        self.eob_multi
            .resize_with(EOB_MULTI_TOTAL_CONTEXTS, Vec::new);
        for (ctx, cdf) in self.eob_multi.iter_mut().enumerate() {
            *cdf = default_eob_multi_for_tx_size_idx(ctx / EOB_PLANE_COUNT);
        }
    }

    /// Get partition CDF for a context.
    #[must_use]
    pub fn get_partition_cdf(&self, ctx: usize) -> &[u16; 11] {
        &self.partition[ctx.min(PARTITION_CONTEXTS - 1)]
    }

    /// Get mutable partition CDF for a context.
    pub fn get_partition_cdf_mut(&mut self, ctx: usize) -> &mut [u16; 11] {
        &mut self.partition[ctx.min(PARTITION_CONTEXTS - 1)]
    }

    /// Get Y mode CDF for a context.
    #[must_use]
    pub fn get_y_mode_cdf(&self, ctx: usize) -> &[u16; 14] {
        &self.y_mode[ctx.min(INTRA_Y_MODE_CONTEXTS - 1)]
    }

    /// Get skip CDF for a context.
    #[must_use]
    pub fn get_skip_cdf(&self, ctx: usize) -> &[u16; 3] {
        &self.skip[ctx.min(SKIP_CONTEXTS - 1)]
    }

    /// Get EOB multi CDF for a context.
    ///
    /// The context is `tx_size_idx * EOB_PLANE_COUNT + plane`. Indices that
    /// fall outside `EOB_MULTI_TOTAL_CONTEXTS` are clamped to the final
    /// slot, which holds the largest-area default CDF (see
    /// `default_eob_multi_for_tx_size_idx`).
    #[must_use]
    pub fn get_eob_multi_cdf(&self, ctx: usize) -> &[u16] {
        let idx = ctx.min(self.eob_multi.len().saturating_sub(1));
        // The Vec is always populated in `new()`/`reset()`, but defend
        // against degenerate states without unwrap.
        match self.eob_multi.get(idx) {
            Some(cdf) => cdf.as_slice(),
            None => &DEFAULT_EOB_MULTI_16,
        }
    }

    /// Get coefficient base CDF for a context.
    #[must_use]
    pub fn get_coeff_base_cdf(&self, ctx: usize) -> &[u16] {
        if ctx < self.coeff_base.len() {
            &self.coeff_base[ctx]
        } else {
            &DEFAULT_COEFF_BASE_CDF
        }
    }

    /// Get coefficient base EOB CDF for a context.
    #[must_use]
    pub fn get_coeff_base_eob_cdf(&self, ctx: usize) -> &[u16] {
        // For EOB position, use the same as coeff_base
        self.get_coeff_base_cdf(ctx)
    }

    /// Get coefficient BR (base range) CDF for a context.
    #[must_use]
    pub fn get_coeff_br_cdf(&self, ctx: usize) -> &[u16] {
        if ctx < self.coeff_br.len() {
            &self.coeff_br[ctx]
        } else {
            &DEFAULT_COEFF_BR_CDF
        }
    }

    /// Get DC sign CDF for a context.
    #[must_use]
    pub fn get_dc_sign_cdf(&self, ctx: usize) -> &[u16] {
        &self.dc_sign[ctx.min(DC_SIGN_CTX_COUNT - 1)]
    }

    // Mutable versions for updating CDFs during decoding

    /// Get mutable EOB multi CDF for a context.
    ///
    /// Mirrors [`Self::get_eob_multi_cdf`] but yields a mutable slice so
    /// the arithmetic coder can adapt the CDF after each symbol. Returns a
    /// slice into the largest-area slot when `ctx` is out of range.
    pub fn get_eob_multi_cdf_mut(&mut self, ctx: usize) -> &mut [u16] {
        if self.eob_multi.is_empty() {
            // Guarantee the invariant: rebuild from defaults rather than
            // returning an empty slice that would crash arithmetic coding.
            self.eob_multi = (0..EOB_MULTI_TOTAL_CONTEXTS)
                .map(|c| default_eob_multi_for_tx_size_idx(c / EOB_PLANE_COUNT))
                .collect();
        }
        let len = self.eob_multi.len();
        let idx = ctx.min(len - 1);
        // `idx < len` after the clamp above, so the indexed access is safe.
        // We use direct indexing because `get_mut` returns `Option<&mut Vec>`
        // which complicates the slice return without unwrap; the bound is
        // already enforced.
        self.eob_multi[idx].as_mut_slice()
    }

    /// Get mutable coefficient base CDF for a context.
    pub fn get_coeff_base_cdf_mut(&mut self, ctx: usize) -> &mut [u16] {
        if ctx < self.coeff_base.len() {
            &mut self.coeff_base[ctx]
        } else if !self.coeff_base.is_empty() {
            &mut self.coeff_base[0]
        } else {
            &mut []
        }
    }

    /// Get mutable coefficient base EOB CDF for a context.
    pub fn get_coeff_base_eob_cdf_mut(&mut self, ctx: usize) -> &mut [u16] {
        self.get_coeff_base_cdf_mut(ctx)
    }

    /// Get mutable coefficient BR (base range) CDF for a context.
    pub fn get_coeff_br_cdf_mut(&mut self, ctx: usize) -> &mut [u16] {
        if ctx < self.coeff_br.len() {
            &mut self.coeff_br[ctx]
        } else if !self.coeff_br.is_empty() {
            &mut self.coeff_br[0]
        } else {
            &mut []
        }
    }

    /// Get mutable DC sign CDF for a context.
    pub fn get_dc_sign_cdf_mut(&mut self, ctx: usize) -> &mut [u16] {
        let idx = ctx.min(DC_SIGN_CTX_COUNT - 1);
        &mut self.dc_sign[idx]
    }
}

impl Default for CdfContext {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_uniform_cdf() {
        let cdf = create_uniform_cdf(4);
        assert_eq!(cdf.len(), 5);
        assert_eq!(cdf[0], 8192);
        assert_eq!(cdf[1], 16384);
        assert_eq!(cdf[2], 24576);
        assert_eq!(cdf[3], 32768);
        assert_eq!(cdf[4], 0); // Count
    }

    #[test]
    fn test_update_cdf() {
        let mut cdf = create_uniform_cdf(4);

        // Update with symbol 0 should increase its probability
        let orig_0 = cdf[0];
        update_cdf(&mut cdf, 0);
        assert!(cdf[0] >= orig_0);
    }

    #[test]
    fn test_reset_cdf_uniform() {
        let mut cdf = vec![0u16; 5];
        reset_cdf_uniform(&mut cdf);

        assert_eq!(cdf[0], 8192);
        assert_eq!(cdf[3], 32768);
        assert_eq!(cdf[4], 0);
    }

    #[test]
    fn test_copy_cdf() {
        let src = create_uniform_cdf(4);
        let mut dst = vec![0u16; 5];

        copy_cdf(&mut dst, &src);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_is_valid_cdf() {
        let valid_cdf = create_uniform_cdf(4);
        assert!(is_valid_cdf(&valid_cdf));

        let invalid_cdf = vec![100u16, 50, 200, 32768, 0]; // Not monotonic
        assert!(!is_valid_cdf(&invalid_cdf));
    }

    #[test]
    fn test_cdf_context_new() {
        let ctx = CdfContext::new();
        assert_eq!(ctx.partition.len(), PARTITION_CONTEXTS);
        assert_eq!(ctx.y_mode.len(), INTRA_Y_MODE_CONTEXTS);
    }

    #[test]
    fn test_cdf_context_reset() {
        let mut ctx = CdfContext::new();
        ctx.partition[0][0] = 12345;

        ctx.reset();
        assert_eq!(ctx.partition[0], DEFAULT_PARTITION_CDFS[0]);
    }

    #[test]
    fn test_get_partition_cdf() {
        let ctx = CdfContext::new();

        let cdf = ctx.get_partition_cdf(0);
        assert_eq!(cdf, &DEFAULT_PARTITION_CDF_0);

        // Out of bounds should clamp
        let cdf_clamped = ctx.get_partition_cdf(100);
        assert_eq!(cdf_clamped, &DEFAULT_PARTITION_CDF_3);
    }

    #[test]
    fn test_default_cdfs_valid() {
        // Check that default CDFs are valid
        for cdf in &DEFAULT_PARTITION_CDFS {
            assert!(is_valid_cdf(cdf));
        }

        for cdf in &DEFAULT_Y_MODE_CDFS {
            assert!(is_valid_cdf(cdf));
        }

        assert!(is_valid_cdf(&DEFAULT_MV_JOINT_CDF));
        assert!(is_valid_cdf(&DEFAULT_MV_CLASS_CDF));
    }

    #[test]
    fn test_cdf_constants() {
        assert_eq!(CDF_PROB_BITS, 15);
        assert_eq!(CDF_PROB_TOP, 32768);
        assert_eq!(CDF_MAX_COUNT, 32);
    }

    #[test]
    fn test_partition_contexts() {
        assert_eq!(PARTITION_CONTEXTS, 4);
        assert_eq!(PARTITION_TYPES, 10);
    }

    #[test]
    fn test_intra_mode_contexts() {
        assert_eq!(INTRA_MODES, 13);
        assert_eq!(INTRA_Y_MODE_CONTEXTS, 4);
    }

    #[test]
    fn test_mv_constants() {
        assert_eq!(MV_JOINTS, 4);
        assert_eq!(MV_CLASSES, 11);
        assert_eq!(MV_OFFSET_BITS, 10);
    }

    // -------------------------------------------------------------------
    // EOB multi-symbol CDF context routing tests
    // -------------------------------------------------------------------

    /// Tx-size indices match the `TxSize` enum order (0..=18).
    const TX_4X4: usize = 0;
    const TX_8X8: usize = 1;
    const TX_16X16: usize = 2;
    const TX_32X32: usize = 3;
    const TX_64X64: usize = 4;
    const TX_4X8: usize = 5;
    const TX_8X4: usize = 6;
    const TX_8X16: usize = 7;
    const TX_16X8: usize = 8;
    const TX_16X32: usize = 9;
    const TX_32X16: usize = 10;
    const TX_4X16: usize = 13;
    const TX_16X4: usize = 14;

    fn eob_ctx_for(tx_size_idx: usize, plane: usize) -> usize {
        tx_size_idx * EOB_PLANE_COUNT + plane
    }

    #[test]
    fn test_eob_multi_total_contexts() {
        // 19 transform sizes × 3 plane types = 57 contexts.
        assert_eq!(EOB_MULTI_TOTAL_CONTEXTS, 57);
        assert_eq!(TX_SIZE_COUNT, 19);
        assert_eq!(EOB_PLANE_COUNT, 3);
    }

    #[test]
    fn test_eob_multi_vec_populated() {
        let ctx = CdfContext::new();
        assert_eq!(ctx.eob_multi.len(), EOB_MULTI_TOTAL_CONTEXTS);
        for cdf in &ctx.eob_multi {
            assert!(
                cdf.len() == DEFAULT_EOB_MULTI_8.len() || cdf.len() == DEFAULT_EOB_MULTI_16.len(),
                "EOB CDF must be 9 or 17 entries, got {}",
                cdf.len()
            );
        }
    }

    #[test]
    fn test_eob_multi_cdf_length_small_blocks() {
        // Areas ≤ 64 (4×4, 4×8, 8×4, 8×8, 4×16, 16×4) use the 9-entry CDF.
        let ctx = CdfContext::new();
        for plane in 0..EOB_PLANE_COUNT {
            for tx in [TX_4X4, TX_4X8, TX_8X4, TX_8X8, TX_4X16, TX_16X4] {
                let cdf = ctx.get_eob_multi_cdf(eob_ctx_for(tx, plane));
                assert_eq!(
                    cdf.len(),
                    DEFAULT_EOB_MULTI_8.len(),
                    "tx_size_idx={tx} plane={plane} expected 9-entry CDF"
                );
                assert_eq!(cdf, &DEFAULT_EOB_MULTI_8[..]);
            }
        }
    }

    #[test]
    fn test_eob_multi_cdf_length_large_blocks() {
        // Areas ≥ 128 use the 17-entry CDF.
        let ctx = CdfContext::new();
        for plane in 0..EOB_PLANE_COUNT {
            for tx in [
                TX_8X16, TX_16X8, TX_16X16, TX_16X32, TX_32X16, TX_32X32, TX_64X64,
            ] {
                let cdf = ctx.get_eob_multi_cdf(eob_ctx_for(tx, plane));
                assert_eq!(
                    cdf.len(),
                    DEFAULT_EOB_MULTI_16.len(),
                    "tx_size_idx={tx} plane={plane} expected 17-entry CDF"
                );
                assert_eq!(cdf, &DEFAULT_EOB_MULTI_16[..]);
            }
        }
    }

    #[test]
    fn test_eob_multi_cdf_out_of_range_clamps() {
        let ctx = CdfContext::new();
        // ctx values ≥ EOB_MULTI_TOTAL_CONTEXTS clamp to the last slot.
        let clamped = ctx.get_eob_multi_cdf(EOB_MULTI_TOTAL_CONTEXTS + 100);
        let last = ctx.get_eob_multi_cdf(EOB_MULTI_TOTAL_CONTEXTS - 1);
        assert_eq!(clamped, last);
    }

    #[test]
    fn test_eob_multi_cdf_adapts_via_mut() {
        // Mutate via the mutable getter and verify persistence through the
        // immutable getter. This is what the arithmetic coder relies on.
        let mut ctx = CdfContext::new();
        let key = eob_ctx_for(TX_4X4, 0);

        let initial = ctx.get_eob_multi_cdf(key).to_vec();
        {
            let cdf_mut = ctx.get_eob_multi_cdf_mut(key);
            assert_eq!(cdf_mut.len(), DEFAULT_EOB_MULTI_8.len());
            let len = cdf_mut.len();
            // Bump every non-terminator entry; terminator is the count.
            for v in &mut cdf_mut[..len - 1] {
                *v = v.saturating_add(1);
            }
            cdf_mut[len - 1] = 5; // simulate adaptation count
        }
        let after = ctx.get_eob_multi_cdf(key).to_vec();
        assert_ne!(
            initial, after,
            "mutation must be visible via immutable getter"
        );
        assert_eq!(after.len(), initial.len());
        assert_eq!(after[after.len() - 1], 5);
    }

    #[test]
    fn test_eob_multi_distinct_slots_per_plane() {
        // Mutating one (tx_size, plane) must not affect a different plane.
        let mut ctx = CdfContext::new();
        let key_luma = eob_ctx_for(TX_16X16, 0);
        let key_chroma = eob_ctx_for(TX_16X16, 1);

        let chroma_before = ctx.get_eob_multi_cdf(key_chroma).to_vec();
        {
            let luma_mut = ctx.get_eob_multi_cdf_mut(key_luma);
            luma_mut[0] = 42;
        }
        let chroma_after = ctx.get_eob_multi_cdf(key_chroma).to_vec();
        let luma_after = ctx.get_eob_multi_cdf(key_luma).to_vec();

        assert_eq!(chroma_before, chroma_after, "chroma slot untouched");
        assert_eq!(luma_after[0], 42, "luma mutation visible");
    }

    #[test]
    fn test_eob_multi_reset_restores_defaults() {
        let mut ctx = CdfContext::new();
        let key = eob_ctx_for(TX_32X32, 2);
        {
            let cdf_mut = ctx.get_eob_multi_cdf_mut(key);
            cdf_mut[0] = 99;
            cdf_mut[1] = 100;
        }
        ctx.reset();
        let after = ctx.get_eob_multi_cdf(key);
        assert_eq!(
            after,
            &DEFAULT_EOB_MULTI_16[..],
            "reset must rebuild large-block default"
        );
    }

    #[test]
    fn test_eob_multi_cdf_mut_recovers_after_clear() {
        // Defensive: if external code drains `eob_multi`, the mutable
        // getter must rebuild the table rather than panic / unwrap.
        let mut ctx = CdfContext::new();
        ctx.eob_multi.clear();
        let cdf = ctx.get_eob_multi_cdf_mut(0);
        assert!(!cdf.is_empty());
        assert_eq!(ctx.eob_multi.len(), EOB_MULTI_TOTAL_CONTEXTS);
    }

    #[test]
    fn test_eob_multi_default_cdfs_are_valid() {
        // The EOB multi defaults must be monotonic and terminate at the
        // probability cap.
        assert!(is_valid_cdf(&DEFAULT_EOB_MULTI_2));
        assert!(is_valid_cdf(&DEFAULT_EOB_MULTI_4));
        assert!(is_valid_cdf(&DEFAULT_EOB_MULTI_8));
        assert!(is_valid_cdf(&DEFAULT_EOB_MULTI_16));
    }
}
