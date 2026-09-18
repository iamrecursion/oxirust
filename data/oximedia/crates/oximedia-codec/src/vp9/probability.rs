//! VP9 Probability tables.
//!
//! This module provides default probability tables and structures for
//! entropy coding in VP9. These probabilities are used by the boolean
//! decoder to decode various syntax elements.

#![forbid(unsafe_code)]
#![allow(dead_code)]

use super::mv::{MV_CLASS0_SIZE, MV_CLASSES, MV_FP_SIZE, MV_JOINTS, MV_OFFSET_BITS};
use super::partition::{PARTITION_TYPES, TX_SIZES};

/// Number of coefficient probability bands.
pub const COEF_BANDS: usize = 6;

/// Number of coefficient contexts.
pub const COEF_CONTEXTS: usize = 6;

/// Number of unconstrained nodes in coefficient tree.
pub const UNCONSTRAINED_NODES: usize = 3;

/// Number of inter modes.
pub const INTER_MODES: usize = 4;

/// Number of inter mode contexts.
pub const INTER_MODE_CONTEXTS: usize = 7;

/// Number of intra modes.
pub const INTRA_MODES: usize = 10;

/// Number of skip contexts.
pub const SKIP_CONTEXTS: usize = 3;

/// Number of is_inter contexts.
pub const IS_INTER_CONTEXTS: usize = 4;

/// Number of compound mode contexts.
pub const COMP_MODE_CONTEXTS: usize = 5;

/// Number of reference contexts.
#[allow(dead_code)]
pub const REF_CONTEXTS: usize = 5;

/// Number of single reference contexts.
pub const SINGLE_REF_CONTEXTS: usize = 5;

/// Number of compound reference contexts.
pub const COMP_REF_CONTEXTS: usize = 5;

/// Number of planes (Y, U, V).
pub const PLANES: usize = 3;

/// Probability type (0-255, where 128 = 50%).
pub type Prob = u8;

/// Default inter mode probabilities.
const DEFAULT_INTER_MODE_PROBS: [[Prob; INTER_MODES - 1]; INTER_MODE_CONTEXTS] = [
    [2, 173, 34],
    [7, 145, 85],
    [7, 166, 63],
    [7, 94, 66],
    [8, 64, 46],
    [17, 81, 31],
    [25, 29, 30],
];

/// Default intra inter probabilities.
const DEFAULT_INTRA_INTER_PROBS: [Prob; IS_INTER_CONTEXTS] = [9, 102, 187, 225];

/// Default compound mode probabilities.
const DEFAULT_COMP_MODE_PROBS: [Prob; COMP_MODE_CONTEXTS] = [239, 183, 119, 96, 41];

/// Default single reference probabilities.
const DEFAULT_SINGLE_REF_PROBS: [[Prob; 2]; SINGLE_REF_CONTEXTS] =
    [[33, 16], [77, 74], [142, 142], [172, 170], [238, 247]];

/// Default compound reference probabilities.
const DEFAULT_COMP_REF_PROBS: [Prob; COMP_REF_CONTEXTS] = [50, 126, 123, 221, 226];

/// Default skip probabilities.
const DEFAULT_SKIP_PROBS: [Prob; SKIP_CONTEXTS] = [192, 128, 64];

/// Default partition probabilities.
const DEFAULT_PARTITION_PROBS: [[Prob; PARTITION_TYPES - 1]; 16] = [
    [199, 122, 141],
    [147, 63, 159],
    [148, 133, 118],
    [121, 104, 114],
    [174, 73, 87],
    [92, 41, 83],
    [82, 99, 50],
    [53, 39, 39],
    [177, 58, 59],
    [68, 26, 63],
    [52, 79, 25],
    [17, 14, 12],
    [222, 34, 30],
    [72, 16, 44],
    [58, 32, 12],
    [10, 7, 6],
];

/// Default Y mode probabilities for keyframes.
const DEFAULT_KF_Y_MODE_PROBS: [[[Prob; INTRA_MODES - 1]; INTRA_MODES]; INTRA_MODES] = [
    [
        [137, 30, 42, 148, 151, 207, 70, 52, 91],
        [92, 45, 102, 136, 116, 180, 74, 90, 100],
        [73, 32, 19, 187, 222, 215, 46, 34, 100],
        [91, 30, 32, 116, 121, 186, 93, 86, 94],
        [72, 35, 36, 149, 68, 206, 68, 63, 105],
        [73, 31, 28, 138, 57, 124, 55, 122, 151],
        [67, 23, 21, 140, 126, 197, 40, 37, 171],
        [86, 27, 28, 128, 154, 212, 45, 43, 53],
        [74, 32, 27, 107, 86, 160, 63, 134, 102],
        [59, 67, 44, 140, 161, 202, 78, 67, 119],
    ],
    [
        [63, 36, 126, 146, 123, 158, 60, 90, 96],
        [43, 46, 168, 134, 107, 128, 69, 142, 92],
        [44, 29, 68, 159, 201, 177, 50, 57, 77],
        [58, 38, 76, 114, 97, 172, 78, 133, 92],
        [46, 41, 76, 140, 63, 184, 69, 112, 57],
        [38, 32, 85, 140, 46, 112, 54, 151, 133],
        [39, 27, 61, 131, 110, 175, 44, 75, 136],
        [52, 30, 74, 113, 130, 175, 51, 64, 58],
        [47, 35, 80, 100, 74, 143, 64, 163, 74],
        [36, 61, 116, 114, 128, 162, 80, 125, 82],
    ],
    [
        [82, 26, 26, 171, 208, 204, 44, 32, 105],
        [55, 44, 68, 166, 179, 192, 57, 57, 108],
        [42, 26, 11, 199, 241, 228, 23, 15, 85],
        [68, 42, 19, 131, 160, 199, 55, 52, 83],
        [58, 50, 25, 139, 115, 232, 39, 52, 118],
        [50, 35, 33, 153, 104, 162, 64, 59, 131],
        [44, 24, 16, 150, 177, 202, 33, 19, 156],
        [55, 27, 12, 153, 203, 218, 26, 27, 49],
        [53, 49, 21, 110, 116, 168, 59, 80, 76],
        [38, 72, 19, 168, 203, 212, 50, 50, 107],
    ],
    [
        [103, 26, 36, 129, 132, 201, 83, 80, 93],
        [59, 38, 83, 112, 103, 162, 98, 136, 90],
        [62, 30, 23, 158, 200, 207, 59, 57, 50],
        [67, 30, 29, 84, 86, 191, 102, 91, 59],
        [60, 32, 33, 112, 71, 220, 64, 89, 104],
        [53, 26, 34, 130, 56, 149, 84, 120, 103],
        [53, 21, 23, 133, 109, 210, 56, 77, 172],
        [77, 19, 29, 112, 142, 228, 55, 66, 36],
        [61, 29, 29, 93, 97, 165, 83, 175, 162],
        [47, 47, 43, 114, 137, 181, 100, 99, 95],
    ],
    [
        [69, 23, 29, 128, 83, 199, 46, 44, 101],
        [53, 40, 55, 139, 69, 183, 61, 80, 110],
        [40, 29, 19, 161, 180, 207, 43, 24, 91],
        [60, 34, 19, 105, 61, 198, 53, 64, 89],
        [52, 31, 22, 158, 40, 209, 58, 62, 89],
        [44, 31, 29, 147, 46, 158, 56, 102, 198],
        [35, 19, 12, 135, 87, 209, 41, 45, 167],
        [55, 25, 21, 118, 95, 215, 38, 39, 66],
        [51, 38, 25, 113, 58, 164, 70, 93, 97],
        [47, 54, 34, 146, 108, 203, 72, 103, 151],
    ],
    [
        [64, 19, 37, 156, 66, 138, 49, 95, 133],
        [46, 27, 80, 150, 55, 124, 55, 121, 135],
        [36, 23, 27, 165, 149, 166, 54, 64, 118],
        [53, 21, 36, 131, 63, 163, 60, 109, 81],
        [40, 26, 35, 154, 40, 185, 51, 97, 123],
        [35, 19, 34, 179, 19, 97, 48, 129, 124],
        [36, 20, 26, 136, 62, 164, 33, 77, 154],
        [45, 18, 32, 130, 90, 157, 40, 79, 91],
        [45, 26, 28, 129, 45, 129, 49, 147, 123],
        [38, 44, 51, 136, 74, 162, 57, 97, 121],
    ],
    [
        [75, 17, 22, 136, 138, 185, 32, 34, 166],
        [56, 39, 58, 133, 117, 173, 48, 53, 187],
        [35, 21, 12, 161, 212, 207, 20, 23, 145],
        [56, 29, 19, 117, 109, 181, 55, 68, 112],
        [47, 29, 17, 153, 64, 220, 59, 51, 114],
        [46, 16, 24, 136, 76, 147, 41, 64, 172],
        [34, 17, 11, 108, 152, 187, 13, 15, 209],
        [51, 24, 14, 115, 133, 209, 32, 26, 104],
        [55, 30, 18, 122, 79, 179, 44, 88, 116],
        [37, 49, 25, 129, 168, 164, 41, 54, 148],
    ],
    [
        [82, 22, 32, 127, 143, 213, 39, 41, 70],
        [62, 44, 61, 123, 105, 189, 48, 57, 64],
        [47, 25, 17, 175, 222, 220, 24, 30, 86],
        [68, 36, 17, 106, 102, 206, 59, 74, 74],
        [57, 39, 23, 151, 68, 216, 55, 63, 58],
        [49, 30, 35, 141, 70, 168, 82, 40, 115],
        [51, 25, 15, 136, 129, 202, 38, 35, 139],
        [68, 26, 16, 111, 141, 215, 29, 28, 28],
        [59, 39, 19, 114, 75, 180, 77, 104, 42],
        [40, 61, 26, 126, 152, 206, 61, 59, 93],
    ],
    [
        [78, 23, 39, 111, 117, 170, 74, 124, 94],
        [48, 34, 86, 101, 92, 146, 78, 179, 134],
        [47, 22, 24, 138, 187, 178, 68, 69, 59],
        [56, 25, 33, 105, 112, 187, 95, 177, 129],
        [48, 31, 27, 114, 63, 183, 82, 116, 56],
        [43, 28, 37, 121, 63, 123, 61, 192, 169],
        [42, 17, 24, 109, 97, 177, 56, 76, 122],
        [58, 18, 28, 105, 139, 182, 70, 92, 63],
        [46, 23, 32, 74, 86, 150, 67, 183, 88],
        [36, 38, 48, 92, 122, 165, 88, 137, 91],
    ],
    [
        [65, 70, 60, 155, 159, 199, 61, 60, 81],
        [44, 78, 115, 132, 119, 173, 71, 112, 93],
        [39, 38, 21, 184, 227, 206, 42, 32, 64],
        [58, 47, 36, 124, 137, 193, 80, 82, 78],
        [49, 50, 35, 144, 95, 205, 63, 78, 59],
        [41, 53, 52, 148, 71, 142, 65, 128, 51],
        [40, 36, 28, 143, 143, 202, 40, 55, 137],
        [52, 34, 29, 129, 183, 227, 42, 35, 43],
        [42, 44, 44, 104, 105, 164, 64, 130, 80],
        [43, 81, 53, 140, 169, 204, 68, 84, 72],
    ],
];

/// Default UV mode probabilities.
const DEFAULT_UV_MODE_PROBS: [[Prob; INTRA_MODES - 1]; INTRA_MODES] = [
    [144, 11, 54, 157, 195, 130, 46, 58, 108],
    [118, 15, 123, 148, 131, 101, 44, 93, 131],
    [113, 12, 23, 188, 226, 142, 26, 32, 125],
    [120, 11, 50, 123, 163, 135, 64, 77, 103],
    [113, 9, 36, 155, 111, 157, 32, 44, 161],
    [116, 9, 55, 176, 76, 96, 37, 61, 149],
    [115, 9, 28, 141, 161, 167, 21, 25, 193],
    [120, 12, 32, 145, 195, 142, 32, 38, 86],
    [116, 12, 64, 120, 140, 125, 49, 115, 121],
    [102, 19, 66, 162, 182, 122, 35, 59, 128],
];

/// Motion vector probability component.
#[derive(Clone, Debug)]
pub struct MvComponentProbs {
    /// Sign probability.
    pub sign: Prob,
    /// Class probabilities.
    pub classes: [Prob; MV_CLASSES - 1],
    /// Class 0 probabilities.
    pub class0: [Prob; MV_CLASS0_SIZE - 1],
    /// Offset bits probabilities.
    pub bits: [Prob; MV_OFFSET_BITS],
    /// Class 0 fractional precision.
    pub class0_fp: [[Prob; MV_FP_SIZE - 1]; MV_CLASS0_SIZE],
    /// Fractional precision.
    pub fp: [Prob; MV_FP_SIZE - 1],
    /// Class 0 high precision.
    pub class0_hp: Prob,
    /// High precision.
    pub hp: Prob,
}

impl Default for MvComponentProbs {
    fn default() -> Self {
        Self {
            sign: 128,
            classes: [224, 144, 192, 168, 192, 176, 192, 198, 198, 245],
            class0: [216],
            bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
            class0_fp: [[128, 128, 64], [96, 112, 64]],
            fp: [64, 96, 64],
            class0_hp: 160,
            hp: 128,
        }
    }
}

impl MvComponentProbs {
    /// Creates default horizontal component probabilities.
    #[must_use]
    pub fn default_horizontal() -> Self {
        Self {
            sign: 128,
            classes: [216, 128, 176, 160, 176, 176, 192, 198, 198, 245],
            class0: [208],
            bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
            class0_fp: [[128, 128, 64], [96, 112, 64]],
            fp: [64, 96, 64],
            class0_hp: 160,
            hp: 128,
        }
    }

    /// Creates default vertical component probabilities.
    #[must_use]
    pub fn default_vertical() -> Self {
        Self::default()
    }
}

/// Motion vector probabilities for a component.
#[derive(Clone, Debug, Default)]
pub struct MvProbs {
    /// Joint probabilities.
    pub joints: [Prob; MV_JOINTS - 1],
    /// Horizontal component probabilities.
    pub comps: [MvComponentProbs; 2],
}

impl MvProbs {
    /// Creates new motion vector probabilities with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            joints: [32, 64, 96],
            comps: [
                MvComponentProbs::default_vertical(),
                MvComponentProbs::default_horizontal(),
            ],
        }
    }
}

/// Coefficient probabilities for a plane/transform size.
pub type CoefProbs = [[[[Prob; UNCONSTRAINED_NODES]; COEF_CONTEXTS]; COEF_BANDS]; 2];

/// Inter mode probabilities.
pub type InterModeProbs = [[Prob; INTER_MODES - 1]; INTER_MODE_CONTEXTS];

/// Partition probabilities.
pub type PartitionProbs = [[Prob; PARTITION_TYPES - 1]; 16];

/// VP9 probability context.
#[derive(Clone, Debug)]
pub struct ProbabilityContext {
    /// Partition probabilities.
    pub partition: PartitionProbs,
    /// Skip probabilities.
    pub skip: [Prob; SKIP_CONTEXTS],
    /// Inter vs intra probabilities.
    pub intra_inter: [Prob; IS_INTER_CONTEXTS],
    /// Compound mode probabilities.
    pub comp_mode: [Prob; COMP_MODE_CONTEXTS],
    /// Single reference probabilities.
    pub single_ref: [[Prob; 2]; SINGLE_REF_CONTEXTS],
    /// Compound reference probabilities.
    pub comp_ref: [Prob; COMP_REF_CONTEXTS],
    /// Inter mode probabilities.
    pub inter_mode: InterModeProbs,
    /// Y mode probabilities (keyframe).
    pub kf_y_mode: [[[Prob; INTRA_MODES - 1]; INTRA_MODES]; INTRA_MODES],
    /// UV mode probabilities.
    pub uv_mode: [[Prob; INTRA_MODES - 1]; INTRA_MODES],
    /// Motion vector probabilities.
    pub mv: MvProbs,
    /// Coefficient probabilities per plane and TX size.
    pub coef: [[CoefProbs; PLANES]; TX_SIZES],
}

impl Default for ProbabilityContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbabilityContext {
    /// Creates a new probability context with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            partition: DEFAULT_PARTITION_PROBS,
            skip: DEFAULT_SKIP_PROBS,
            intra_inter: DEFAULT_INTRA_INTER_PROBS,
            comp_mode: DEFAULT_COMP_MODE_PROBS,
            single_ref: DEFAULT_SINGLE_REF_PROBS,
            comp_ref: DEFAULT_COMP_REF_PROBS,
            inter_mode: DEFAULT_INTER_MODE_PROBS,
            kf_y_mode: DEFAULT_KF_Y_MODE_PROBS,
            uv_mode: DEFAULT_UV_MODE_PROBS,
            mv: MvProbs::new(),
            coef: Self::default_coef_probs(),
        }
    }

    /// Creates default coefficient probabilities.
    fn default_coef_probs() -> [[CoefProbs; PLANES]; TX_SIZES] {
        // Initialize with reasonable defaults
        let default_node: [Prob; UNCONSTRAINED_NODES] = [128, 128, 128];
        let default_context: [[Prob; UNCONSTRAINED_NODES]; COEF_CONTEXTS] =
            [default_node; COEF_CONTEXTS];
        let default_band: [[[Prob; UNCONSTRAINED_NODES]; COEF_CONTEXTS]; COEF_BANDS] =
            [default_context; COEF_BANDS];
        let default_coef: CoefProbs = [default_band; 2];
        let default_plane: [CoefProbs; PLANES] = [default_coef; PLANES];
        [default_plane; TX_SIZES]
    }

    /// Resets all probabilities to defaults.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Updates a partition probability.
    pub fn update_partition(&mut self, ctx: usize, idx: usize, prob: Prob) {
        if ctx < 16 && idx < PARTITION_TYPES - 1 {
            self.partition[ctx][idx] = prob;
        }
    }

    /// Updates a skip probability.
    pub fn update_skip(&mut self, ctx: usize, prob: Prob) {
        if ctx < SKIP_CONTEXTS {
            self.skip[ctx] = prob;
        }
    }

    /// Updates an intra/inter probability.
    pub fn update_intra_inter(&mut self, ctx: usize, prob: Prob) {
        if ctx < IS_INTER_CONTEXTS {
            self.intra_inter[ctx] = prob;
        }
    }

    /// Updates an inter mode probability.
    pub fn update_inter_mode(&mut self, ctx: usize, idx: usize, prob: Prob) {
        if ctx < INTER_MODE_CONTEXTS && idx < INTER_MODES - 1 {
            self.inter_mode[ctx][idx] = prob;
        }
    }

    /// Returns the partition probability for a context.
    #[must_use]
    pub fn get_partition_probs(&self, ctx: usize) -> &[Prob; PARTITION_TYPES - 1] {
        &self.partition[ctx.min(15)]
    }

    /// Returns the skip probability for a context.
    #[must_use]
    pub fn get_skip_prob(&self, ctx: usize) -> Prob {
        self.skip[ctx.min(SKIP_CONTEXTS - 1)]
    }

    /// Returns the intra/inter probability for a context.
    #[must_use]
    pub fn get_intra_inter_prob(&self, ctx: usize) -> Prob {
        self.intra_inter[ctx.min(IS_INTER_CONTEXTS - 1)]
    }

    /// Returns the inter mode probabilities for a context.
    #[must_use]
    pub fn get_inter_mode_probs(&self, ctx: usize) -> &[Prob; INTER_MODES - 1] {
        &self.inter_mode[ctx.min(INTER_MODE_CONTEXTS - 1)]
    }

    /// Returns Y mode probabilities for keyframe.
    #[must_use]
    pub fn get_kf_y_mode_probs(&self, above: usize, left: usize) -> &[Prob; INTRA_MODES - 1] {
        &self.kf_y_mode[above.min(INTRA_MODES - 1)][left.min(INTRA_MODES - 1)]
    }

    /// Returns UV mode probabilities.
    #[must_use]
    pub fn get_uv_mode_probs(&self, y_mode: usize) -> &[Prob; INTRA_MODES - 1] {
        &self.uv_mode[y_mode.min(INTRA_MODES - 1)]
    }
}

/// Probability update helpers.
pub mod update {
    use super::Prob;

    /// Number of update bits for probabilities.
    pub const PROB_UPDATE_BITS: u32 = 8;

    /// Merges probabilities using weighted average.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn merge_prob(pre_prob: Prob, update_prob: Prob, update_factor: u8) -> Prob {
        let factor = u32::from(update_factor);
        let pre = u32::from(pre_prob);
        let upd = u32::from(update_prob);

        let result = pre + (((upd - pre) * factor + 128) >> 8);
        result.clamp(1, 255) as Prob
    }

    /// Inverts a probability.
    #[must_use]
    pub const fn invert_prob(prob: Prob) -> Prob {
        255 - prob
    }

    /// Returns the branch count weight.
    #[must_use]
    pub const fn branch_weight(count: u32) -> u8 {
        match count {
            0..=15 => 24,
            16..=31 => 48,
            32..=63 => 56,
            _ => 64,
        }
    }
}

/// Frame context storage for probability adaptation.
#[derive(Clone, Debug, Default)]
pub struct FrameContext {
    /// Probability context.
    pub probs: ProbabilityContext,
    /// Frame counts for adaptation.
    pub counts: FrameCounts,
}

impl FrameContext {
    /// Creates a new frame context with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            probs: ProbabilityContext::new(),
            counts: FrameCounts::default(),
        }
    }

    /// Adapts probabilities based on counts.
    pub fn adapt(&mut self) {
        // Adapt skip probabilities
        for (ctx, (prob, count)) in self
            .probs
            .skip
            .iter_mut()
            .zip(self.counts.skip.iter())
            .enumerate()
        {
            let _ = ctx; // Silence unused warning
            let total = count.0 + count.1;
            if total > 0 {
                let weight = update::branch_weight(total);
                #[allow(clippy::cast_possible_truncation)]
                let new_prob = ((count.0 * 256) / total).clamp(1, 255) as Prob;
                *prob = update::merge_prob(*prob, new_prob, weight);
            }
        }

        // Reset counts for next frame
        self.counts.reset();
    }

    /// Resets the context.
    pub fn reset(&mut self) {
        self.probs.reset();
        self.counts.reset();
    }
}

/// Frame counts for probability adaptation.
#[derive(Clone, Debug, Default)]
pub struct FrameCounts {
    /// Skip counts.
    pub skip: [(u32, u32); SKIP_CONTEXTS],
    /// Intra/inter counts.
    pub intra_inter: [(u32, u32); IS_INTER_CONTEXTS],
    /// Partition counts.
    pub partition: [[(u32, u32, u32, u32); PARTITION_TYPES]; 16],
}

impl FrameCounts {
    /// Resets all counts to zero.
    pub fn reset(&mut self) {
        self.skip = [(0, 0); SKIP_CONTEXTS];
        self.intra_inter = [(0, 0); IS_INTER_CONTEXTS];
        self.partition = [[(0, 0, 0, 0); PARTITION_TYPES]; 16];
    }

    /// Increments skip count.
    pub fn count_skip(&mut self, ctx: usize, skip: bool) {
        if ctx < SKIP_CONTEXTS {
            if skip {
                self.skip[ctx].0 += 1;
            } else {
                self.skip[ctx].1 += 1;
            }
        }
    }

    /// Increments intra/inter count.
    pub fn count_intra_inter(&mut self, ctx: usize, is_inter: bool) {
        if ctx < IS_INTER_CONTEXTS {
            if is_inter {
                self.intra_inter[ctx].1 += 1;
            } else {
                self.intra_inter[ctx].0 += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability_context_new() {
        let ctx = ProbabilityContext::new();
        assert_eq!(ctx.skip.len(), SKIP_CONTEXTS);
        assert_eq!(ctx.intra_inter.len(), IS_INTER_CONTEXTS);
    }

    #[test]
    fn test_partition_probs() {
        let ctx = ProbabilityContext::new();
        let probs = ctx.get_partition_probs(0);
        assert_eq!(probs.len(), PARTITION_TYPES - 1);
    }

    #[test]
    fn test_skip_probs() {
        let ctx = ProbabilityContext::new();
        let prob = ctx.get_skip_prob(0);
        assert_eq!(prob, 192);
    }

    #[test]
    fn test_inter_mode_probs() {
        let ctx = ProbabilityContext::new();
        let probs = ctx.get_inter_mode_probs(0);
        assert_eq!(probs.len(), INTER_MODES - 1);
    }

    #[test]
    fn test_kf_y_mode_probs() {
        let ctx = ProbabilityContext::new();
        let probs = ctx.get_kf_y_mode_probs(0, 0);
        assert_eq!(probs.len(), INTRA_MODES - 1);
    }

    #[test]
    fn test_uv_mode_probs() {
        let ctx = ProbabilityContext::new();
        let probs = ctx.get_uv_mode_probs(0);
        assert_eq!(probs.len(), INTRA_MODES - 1);
    }

    #[test]
    fn test_mv_probs() {
        let probs = MvProbs::new();
        assert_eq!(probs.joints.len(), MV_JOINTS - 1);
        assert_eq!(probs.comps.len(), 2);
    }

    #[test]
    fn test_mv_component_probs() {
        let h = MvComponentProbs::default_horizontal();
        let v = MvComponentProbs::default_vertical();
        assert_eq!(h.classes.len(), MV_CLASSES - 1);
        assert_eq!(v.bits.len(), MV_OFFSET_BITS);
    }

    #[test]
    fn test_update_partition() {
        let mut ctx = ProbabilityContext::new();
        ctx.update_partition(0, 0, 200);
        assert_eq!(ctx.partition[0][0], 200);
    }

    #[test]
    fn test_update_skip() {
        let mut ctx = ProbabilityContext::new();
        ctx.update_skip(1, 100);
        assert_eq!(ctx.skip[1], 100);
    }

    #[test]
    fn test_merge_prob() {
        let result = update::merge_prob(128, 200, 64);
        assert!(result > 128 && result < 200);
    }

    #[test]
    fn test_invert_prob() {
        assert_eq!(update::invert_prob(0), 255);
        assert_eq!(update::invert_prob(255), 0);
        assert_eq!(update::invert_prob(128), 127);
    }

    #[test]
    fn test_branch_weight() {
        assert_eq!(update::branch_weight(0), 24);
        assert_eq!(update::branch_weight(20), 48);
        assert_eq!(update::branch_weight(50), 56);
        assert_eq!(update::branch_weight(100), 64);
    }

    #[test]
    fn test_frame_context() {
        let mut ctx = FrameContext::new();
        ctx.counts.count_skip(0, true);
        ctx.counts.count_skip(0, true);
        ctx.counts.count_skip(0, false);
        assert_eq!(ctx.counts.skip[0], (2, 1));
    }

    #[test]
    fn test_frame_counts_reset() {
        let mut counts = FrameCounts::default();
        counts.count_skip(0, true);
        counts.reset();
        assert_eq!(counts.skip[0], (0, 0));
    }

    #[test]
    fn test_probability_context_reset() {
        let mut ctx = ProbabilityContext::new();
        ctx.update_skip(0, 50);
        ctx.reset();
        assert_eq!(ctx.skip[0], 192);
    }
}
