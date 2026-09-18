//! JPEG-LS RUN mode primitives (ISO/IEC 14495-1 §A.7).
//!
//! When the three quantised gradient differences `d1 = D − B`, `d2 = B − C`
//! and `d3 = C − A` all vanish (lossless) or remain within `±NEAR`
//! (near-lossless), the encoder/decoder pair switches from the §A.6 regular
//! mode into RUN mode: a run of consecutive matching samples is coded as a
//! sequence of one-bit length tokens drawn from the run-length table
//! `J[]` (Table A.5).  When the run is interrupted by a non-matching sample
//! before the end of the current line, the residual length is encoded as
//! a `J[run_index]`-bit suffix and the breaking sample is coded with one
//! of two special "RUN-interruption" contexts (365 if `Ra == Rb`, 366
//! otherwise).  At end-of-line a single `1` bit signals the remaining
//! samples up to the line boundary all match the run value.
//!
//! This module provides the **shared** primitives used by both
//! [`super::encoder`] and [`super::decoder`]: the J table, the threshold
//! lookup `RUN_THRESHOLD`, the per-component [`RunState`] (run index
//! plus the latched run value), the entry tests for the lossless and
//! near-lossless cases, the index bump capped at 30, and the
//! termination-context selector.
//!
//! The numbers themselves come straight from ISO 14495-1 Table A.5 and
//! §A.7.2.  Every constant is reproduced here exactly so the encoder and
//! decoder agree at every transition point without indirection through
//! a runtime-loaded table.

use super::context::NUM_REGULAR_CONTEXTS;

/// ISO 14495-1 Table A.5 — the Golomb-Rice order used inside RUN mode.
///
/// `J[run_index]` is the number of suffix bits emitted for the residual
/// run length when the run is interrupted, and equivalently the binary
/// logarithm of the full-token length `RUN_THRESHOLD[run_index]`.  The
/// 31 entries correspond to indices `0..=30`; the index is capped at 30
/// (see [`bump_run_index`]) so the largest threshold ever reached is
/// `2^15 = 32768`.
pub const J: [i32; 31] = [
    0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14,
    15,
];

/// Pre-computed `1 << J[run_index]` for every legal run index.
///
/// This is the length, in samples, of one full RUN token at index
/// `run_index`.  The encoder emits a single `1` bit when it can consume
/// exactly `RUN_THRESHOLD[run_index]` matching samples from the current
/// column without leaving the row; the decoder mirrors that.
pub const RUN_THRESHOLD: [i32; 31] = [
    1 << 0,
    1 << 0,
    1 << 0,
    1 << 1,
    1 << 1,
    1 << 1,
    1 << 1,
    1 << 2,
    1 << 2,
    1 << 2,
    1 << 2,
    1 << 3,
    1 << 3,
    1 << 3,
    1 << 3,
    1 << 4,
    1 << 4,
    1 << 5,
    1 << 5,
    1 << 6,
    1 << 6,
    1 << 7,
    1 << 7,
    1 << 8,
    1 << 9,
    1 << 10,
    1 << 11,
    1 << 12,
    1 << 13,
    1 << 14,
    1 << 15,
];

/// Largest legal value for [`RunState::run_index`].  Indices are clamped
/// to `MAX_RUN_INDEX` by [`bump_run_index`]; both `J` and
/// `RUN_THRESHOLD` are indexable across `0..=MAX_RUN_INDEX`.
pub const MAX_RUN_INDEX: usize = 30;

/// Index of the first RUN-interruption context.
///
/// Two new contexts (one per RIType) are appended after the 365 regular
/// contexts: `RUN_TERMINATION_CTX_BASE + 0` is selected when the breaking
/// sample's left neighbour equals its top neighbour (`Ra == Rb`,
/// RIType = 0), and `RUN_TERMINATION_CTX_BASE + 1` otherwise
/// (RIType = 1).
pub const RUN_TERMINATION_CTX_BASE: usize = NUM_REGULAR_CONTEXTS;

/// Total number of RUN-interruption contexts (one per RIType).
pub const NUM_RUN_TERMINATION_CONTEXTS: usize = 2;

/// Total number of coding contexts including both regular and RUN
/// termination: 365 + 2 = 367.
pub const TOTAL_CONTEXT_COUNT: usize = NUM_REGULAR_CONTEXTS + NUM_RUN_TERMINATION_CONTEXTS;

/// Per-component RUN-mode state.
///
/// Encoder and decoder allocate one `RunState` per scan component; the
/// state is reset at the start of every line.  `run_value` is the
/// `Ra` value at the moment RUN mode is entered — i.e., the value that
/// each filled sample inside the run will take on the decoder side.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunState {
    /// Index into the `J[]` / `RUN_THRESHOLD[]` tables (`0..=MAX_RUN_INDEX`).
    pub run_index: usize,
    /// Latched left-neighbour value used as the run's repeated sample.
    pub run_value: i32,
}

impl RunState {
    /// Create a fresh RUN state with `run_index = 0` and `run_value = 0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            run_index: 0,
            run_value: 0,
        }
    }

    /// Reset the run index to zero; called at the start of every new line.
    pub fn reset_at_line_start(&mut self) {
        self.run_index = 0;
        // run_value is overwritten on every RUN entry so it does not need a reset.
    }
}

/// Lossless RUN-entry test (ISO 14495-1 §A.7.1, NEAR = 0).
///
/// Returns `true` when all three raw gradient differences vanish.  This is
/// the exact predicate the existing context quantiser maps to
/// `q1 = q2 = q3 = 0` (context index 40), but expressed on the raw
/// differences so encoder and decoder use the *same* test on the
/// *same* data — a slight numerical mismatch between the quantised
/// and raw zero conditions could otherwise desync the pair.
#[inline]
#[must_use]
pub fn enter_run_lossless(d1: i32, d2: i32, d3: i32) -> bool {
    d1 == 0 && d2 == 0 && d3 == 0
}

/// Near-lossless RUN-entry test (ISO 14495-1 §A.7.1, NEAR > 0).
///
/// Returns `true` when each raw gradient difference is within `±near`.
/// At `near = 0` this reduces to [`enter_run_lossless`].
#[inline]
#[must_use]
pub fn enter_run_near(d1: i32, d2: i32, d3: i32, near: i32) -> bool {
    d1.abs() <= near && d2.abs() <= near && d3.abs() <= near
}

/// Advance the run index after a full-length token, clamped at
/// `MAX_RUN_INDEX = 30` per ISO 14495-1 §A.7.2.
///
/// The clamp matches the decoder's clamp, ensuring both sides see the
/// same `run_index` after every full-token emit/decode.
#[inline]
pub fn bump_run_index(state: &mut RunState) {
    if state.run_index < MAX_RUN_INDEX {
        state.run_index += 1;
    }
}

/// Decrement the run index by one (saturating at zero), called after a
/// RUN-interruption sample is emitted/decoded.
#[inline]
pub fn decrement_run_index(state: &mut RunState) {
    if state.run_index > 0 {
        state.run_index -= 1;
    }
}

/// Return the RUN-interruption context index for a breaking sample
/// whose left neighbour is `ra` and whose top neighbour is `rb`.
///
/// Encodes RIType = 0 (Ra == Rb) as [`RUN_TERMINATION_CTX_BASE`] and
/// RIType = 1 (Ra != Rb) as `RUN_TERMINATION_CTX_BASE + 1`.
#[inline]
#[must_use]
pub fn run_termination_ctx(ra: i32, rb: i32) -> usize {
    if ra == rb {
        RUN_TERMINATION_CTX_BASE
    } else {
        RUN_TERMINATION_CTX_BASE + 1
    }
}

/// Return the run-length threshold `1 << J[run_index]` for the given
/// `run_index`, or `0` if the index is out of bounds (defensive: in
/// practice the encoder/decoder keep `run_index <= MAX_RUN_INDEX`).
#[inline]
#[must_use]
pub fn threshold_for(run_index: usize) -> i32 {
    if run_index < RUN_THRESHOLD.len() {
        RUN_THRESHOLD[run_index]
    } else {
        0
    }
}

/// Return `J[run_index]` (the number of suffix bits used for the
/// residual length), or `0` for an out-of-bounds index.
#[inline]
#[must_use]
pub fn j_for(run_index: usize) -> i32 {
    if run_index < J.len() {
        J[run_index]
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn j_table_has_31_entries() {
        assert_eq!(J.len(), 31);
        assert_eq!(RUN_THRESHOLD.len(), 31);
    }

    #[test]
    fn j_table_matches_iso_table_a5() {
        // Spot-check a handful of values against ISO/IEC 14495-1 Table A.5.
        assert_eq!(J[0], 0);
        assert_eq!(J[2], 0);
        assert_eq!(J[3], 1);
        assert_eq!(J[7], 2);
        assert_eq!(J[15], 4);
        assert_eq!(J[16], 4);
        assert_eq!(J[17], 5);
        assert_eq!(J[30], 15);
    }

    #[test]
    fn thresholds_consistent_with_j() {
        for (idx, &j) in J.iter().enumerate() {
            assert_eq!(
                RUN_THRESHOLD[idx],
                1 << j,
                "RUN_THRESHOLD[{idx}] mismatched against 1 << J[{idx}] = 1 << {j}"
            );
        }
    }

    #[test]
    fn enter_run_lossless_only_when_all_zero() {
        assert!(enter_run_lossless(0, 0, 0));
        assert!(!enter_run_lossless(1, 0, 0));
        assert!(!enter_run_lossless(0, 1, 0));
        assert!(!enter_run_lossless(0, 0, 1));
        assert!(!enter_run_lossless(-1, 0, 0));
    }

    #[test]
    fn enter_run_near_respects_near_bound() {
        assert!(enter_run_near(0, 0, 0, 0));
        assert!(enter_run_near(2, -2, 1, 2));
        assert!(!enter_run_near(3, 0, 0, 2));
        assert!(!enter_run_near(0, -3, 0, 2));
        assert!(!enter_run_near(0, 0, 3, 2));
    }

    #[test]
    fn bump_run_index_is_capped_at_max() {
        let mut state = RunState::new();
        for _ in 0..100 {
            bump_run_index(&mut state);
        }
        assert_eq!(state.run_index, MAX_RUN_INDEX);
    }

    #[test]
    fn decrement_run_index_saturates_at_zero() {
        let mut state = RunState::new();
        // Already 0 — decrementing should stay at 0.
        decrement_run_index(&mut state);
        assert_eq!(state.run_index, 0);
        // Bump once, then decrement should bring us back to 0.
        bump_run_index(&mut state);
        assert_eq!(state.run_index, 1);
        decrement_run_index(&mut state);
        assert_eq!(state.run_index, 0);
    }

    #[test]
    fn termination_context_selects_365_or_366() {
        // With NUM_REGULAR_CONTEXTS == 365 the two RUN-interruption contexts
        // are 365 (Ra == Rb) and 366 (Ra != Rb).
        assert_eq!(run_termination_ctx(100, 100), 365);
        assert_eq!(run_termination_ctx(100, 50), 366);
        assert_eq!(run_termination_ctx(0, 1), 366);
        assert_eq!(run_termination_ctx(0, 0), 365);
    }

    #[test]
    fn threshold_for_matches_run_threshold_table() {
        for idx in 0..=MAX_RUN_INDEX {
            assert_eq!(threshold_for(idx), RUN_THRESHOLD[idx]);
        }
        // Out-of-range indices return 0 defensively.
        assert_eq!(threshold_for(MAX_RUN_INDEX + 1), 0);
    }

    #[test]
    fn j_for_matches_j_table() {
        for idx in 0..=MAX_RUN_INDEX {
            assert_eq!(j_for(idx), J[idx]);
        }
        assert_eq!(j_for(MAX_RUN_INDEX + 1), 0);
    }

    #[test]
    fn reset_at_line_start_zeroes_index() {
        let mut state = RunState::new();
        state.run_index = 17;
        state.run_value = 42;
        state.reset_at_line_start();
        assert_eq!(state.run_index, 0);
        // run_value is not asserted because it is overwritten on every entry.
    }

    #[test]
    fn total_context_count_is_367() {
        assert_eq!(TOTAL_CONTEXT_COUNT, 367);
        assert_eq!(RUN_TERMINATION_CTX_BASE, 365);
    }
}
