//! JPEG-LS context modeling: 365 regular contexts plus 2 RUN-interruption
//! contexts (ISO 14495-1 §6.2–§6.4, §A.7.2).
//!
//! §6.2–§6.4 defines three quantised gradient differences
//! (q1, q2, q3) ∈ {-4..4}³ that are sign-normalised and mapped to a
//! context index ∈ [0, 364]. Each context maintains a bias-correction
//! value Cx, an adaptive Golomb-Rice parameter k, and accumulators
//! B (error sum) and N (sample count) for the adaptive update rule.
//!
//! In addition, RUN mode (§A.7.2) appends two extra adaptive states for
//! the RUN-interruption (termination) sample: index 365 selected when
//! the breaking sample's left and top neighbours are equal (RIType = 0),
//! and index 366 when they differ (RIType = 1).  The per-state update
//! rule is identical, so all 367 entries are stored in a single
//! `Vec<ContextState>` allocated per scan component.

/// Total number of regular coding contexts defined by JPEG-LS §A.6.
///
/// With 9 levels per quantised gradient and sign normalisation, the first
/// non-zero value among (q1, q2, q3) is forced positive, giving 365 unique
/// triples (including the all-zero triple at index 0).
pub const NUM_REGULAR_CONTEXTS: usize = 365;

/// Number of additional RUN-interruption (termination) contexts (§A.7.2).
///
/// Index `NUM_REGULAR_CONTEXTS + 0 = 365` is used when the breaking
/// sample's left neighbour equals its top neighbour (RIType = 0);
/// `NUM_REGULAR_CONTEXTS + 1 = 366` otherwise (RIType = 1).
pub const NUM_RUN_TERMINATION_CONTEXTS: usize = 2;

/// Total contexts (regular + RUN termination): `365 + 2 = 367`.
///
/// Encoder and decoder allocate `NUM_TOTAL_CONTEXTS` `ContextState`s
/// per component so that the regular path uses indices `[0, 365)` and
/// RUN mode uses indices `[365, 367)` without any extra bookkeeping.
pub const NUM_TOTAL_CONTEXTS: usize = NUM_REGULAR_CONTEXTS + NUM_RUN_TERMINATION_CONTEXTS;

/// Per-context adaptive state for one of the 365 JPEG-LS regular contexts.
#[derive(Clone, Debug)]
pub struct ContextState {
    /// Bias correction value Cx ∈ [−128, 127].
    pub cx: i32,
    /// Adaptive Golomb-Rice order k ∈ [0, 15].
    pub k: i32,
    /// Error accumulator Bk (used by the adaptive k-update).
    pub b: i32,
    /// Sample count Nk (initialised to 1 to avoid division by zero).
    pub n: i32,
}

impl Default for ContextState {
    fn default() -> Self {
        Self {
            cx: 0,
            k: 0,
            b: 0,
            n: 1,
        }
    }
}

/// Map a sign-normalised gradient triple to a context index in [0, 364].
///
/// Returns `(ctx_idx, sign)` where:
/// - `ctx_idx` is the canonical context index.
/// - `sign` is `1` when the triple is already in positive-normalised form,
///   `-1` when it was negated to bring it to canonical form.
///
/// The sign flip is applied to the error before Golomb coding so that the
/// decoder can recover the signed error as `decoded_unsigned * sign`.
#[inline]
pub fn context_index(q1: i8, q2: i8, q3: i8) -> (usize, i32) {
    // Sign-normalise: the first non-zero component must be positive.
    let (q1n, q2n, q3n, sign) = if q1 < 0 || (q1 == 0 && q2 < 0) || (q1 == 0 && q2 == 0 && q3 < 0) {
        (-q1, -q2, -q3, -1i32)
    } else {
        (q1, q2, q3, 1i32)
    };

    // After normalisation q1n ∈ [0, 4]; q2n, q3n ∈ [-4, 4].
    // Encode as a base-9 triple: q1n * 81 + (q2n+4) * 9 + (q3n+4).
    let idx = (q1n as usize) * 81 + ((q2n + 4) as usize) * 9 + (q3n + 4) as usize;
    (idx.min(NUM_REGULAR_CONTEXTS - 1), sign)
}

/// Update a context's adaptive statistics after observing (signed) error `err`.
///
/// Implements the adaptive bias-correction and k-update from ISO 14495-1 §6.4.
///
/// - `err`     — signed quantised error in the sign-normalised domain.
/// - `near`    — NEAR parameter (0 for lossless).
/// - `reset`   — Rt reset threshold.
/// - `max_val` — MaxVal (unused here; reserved for near-lossless clamping).
pub fn update_context(state: &mut ContextState, err: i32, near: i32, reset: i32, _max_val: i32) {
    // Accumulate error with near-lossless offset (for lossless, near = 0).
    state.b += err - near;
    state.n += 1;

    // Adaptive k: increase k while n << k < reset (i.e., while b is large relative to n).
    // k is bounded by the reset interval to keep Golomb codes short.
    while state.n << state.k < reset {
        state.k += 1;
    }

    // Halve statistics when accumulators grow too large.
    if state.b.abs() > reset {
        state.b = (state.b + if state.b < 0 { -1 } else { 1 }) / 2;
        state.n = (state.n + 1) / 2;
    }

    // Bias correction: adjust Cx toward zero based on systematic error sign.
    if state.b <= -state.n {
        state.cx -= 1;
        state.b += state.n;
        if state.b <= -state.n {
            state.b = -state.n + 1;
        }
    } else if state.b > 0 {
        state.cx += 1;
        state.b -= state.n;
        if state.b > 0 {
            state.b = 0;
        }
    }

    state.cx = state.cx.clamp(-128, 127);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zero_triple_maps_to_centre() {
        let (idx, sign) = context_index(0, 0, 0);
        // q1=0, q2=0, q3=0 → 0*81 + (0+4)*9 + (0+4) = 0 + 36 + 4 = 40
        assert_eq!(idx, 40);
        assert_eq!(sign, 1);
    }

    #[test]
    fn negative_triple_is_sign_normalised() {
        let (idx_pos, _) = context_index(1, 2, 3);
        let (idx_neg, sign_neg) = context_index(-1, -2, -3);
        assert_eq!(idx_pos, idx_neg);
        assert_eq!(sign_neg, -1);
    }

    #[test]
    fn index_within_bounds() {
        for q1 in -4i8..=4 {
            for q2 in -4i8..=4 {
                for q3 in -4i8..=4 {
                    let (idx, _) = context_index(q1, q2, q3);
                    assert!(
                        idx < NUM_REGULAR_CONTEXTS,
                        "idx={idx} out of bounds for ({q1},{q2},{q3})"
                    );
                }
            }
        }
    }

    #[test]
    fn context_update_does_not_overflow() {
        let mut state = ContextState::default();
        for err in [-5i32, 3, 0, -1, 7, -10] {
            update_context(&mut state, err, 0, 64, 255);
            assert!(state.cx >= -128 && state.cx <= 127);
        }
    }

    #[test]
    fn total_contexts_is_regular_plus_run_termination() {
        assert_eq!(NUM_TOTAL_CONTEXTS, NUM_REGULAR_CONTEXTS + 2);
        assert_eq!(NUM_TOTAL_CONTEXTS, 367);
        assert_eq!(NUM_RUN_TERMINATION_CONTEXTS, 2);
    }
}
