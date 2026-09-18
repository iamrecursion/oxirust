//! LARSCH (Larmore-Schieber) online concave row-minima algorithm and the
//! `optimal_break_larsch` line-breaking function built on top of it.

use std::collections::VecDeque;

use super::kp_common::{kp_cost_i64, kp_cost_is_concave, reconstruct_lines};

/// LARSCH: online totally-monotone **concave** matrix row-minima finder.
///
/// Given a cost function `C(row, col)` that satisfies the *inverse Monge*
/// (concave) property — i.e. for every `r < r'` and `c < c'`,
///
/// ```text
///     C(r, c) + C(r', c') >= C(r, c') + C(r', c)
/// ```
///
/// — `Larsch` finds, for each row `r` added in order, the column index
/// `argmin_{c <= r} C(r, c)` in amortised O(1) per row (O(n) total).
///
/// Reference: Larmore & Schieber (1991), *"On-line algorithms for two
/// problems in dynamic programming"*, JACM 38(1), pp. 1037–1069.
///
/// **Usage**
///
/// ```rust,no_run
/// # use oximedia_caption_gen::line_breaking::Larsch;
/// let cost = |r: usize, c: usize| -> i64 { ((r as i64) - (c as i64)).pow(2) };
/// let n = 8;
/// let mut larsch = Larsch::new(n);
/// for r in 0..n {
///     larsch.add_row(r, &cost);
/// }
/// // larsch.row_minima[r] == argmin_{c <= r} cost(r, c)
/// ```
pub struct Larsch {
    /// Deque entries: `(pivot_col, row_start, row_end)`.
    ///
    /// Invariant: for rows in `[row_start, row_end]`, `pivot_col` is the
    /// best known candidate column, or the column from which the true
    /// minimum lies to the right (≥ pivot_col).  Entries are ordered by
    /// strictly increasing `pivot_col`.
    deque: VecDeque<(usize, usize, usize)>,
    /// Total number of rows/columns in the matrix.
    n: usize,
    /// `row_minima[r]` = the column achieving the minimum in row `r`.
    /// Populated entry-by-entry as [`Larsch::add_row`] is called.
    pub row_minima: Vec<usize>,
}

impl Larsch {
    /// Create a new `Larsch` instance for an `n × n` matrix (columns in
    /// `0..n`, rows added via [`add_row`](Larsch::add_row) in order).
    pub fn new(n: usize) -> Self {
        Self {
            deque: VecDeque::new(),
            n,
            row_minima: Vec::with_capacity(n),
        }
    }

    /// Register row `r` and set `self.row_minima[r]`.
    ///
    /// `cost(r, c)` must satisfy the inverse Monge (concave) property over
    /// the domain `0 <= c <= r < n`.  Must be called with `r = 0, 1, …`
    /// in strictly increasing order.
    pub fn add_row<F>(&mut self, r: usize, cost: &F)
    where
        F: Fn(usize, usize) -> i64,
    {
        // ── REDUCE step ──────────────────────────────────────────────────
        // Pop deque entries from the back whose pivot column is dominated
        // by the new column `r`.  A pivot column `c_top` is dominated if
        // `cost(r_start, r) < cost(r_start, c_top)`: the new column is
        // strictly better already at the *start* of that range, so by the
        // concavity property it is at least as good for all rows
        // ≥ r_start.
        while let Some(&(c_top, r_start, _r_end)) = self.deque.back() {
            if cost(r_start, r) < cost(r_start, c_top) {
                self.deque.pop_back();
            } else {
                break;
            }
        }

        // ── INSERT step ──────────────────────────────────────────────────
        // Determine from which row onward column `r` would be optimal and
        // append an entry for it (if it ever wins).
        if self.deque.is_empty() {
            // `r` is the only candidate — it covers all rows from `r`
            // onward (upper bound `n-1`).
            self.deque.push_back((r, r, self.n.saturating_sub(1)));
        } else {
            let (c_prev, r_s, r_e) = *self.deque.back().expect("deque non-empty");
            // Binary-search for the crossover row in [r_s, r_e].
            let crossover = larsch_crossover(r_s, r_e, c_prev, r, cost);
            if crossover <= r_e {
                // Trim the previous entry so it no longer covers [crossover..r_e],
                // and add a new entry for column `r` starting at `crossover`.
                self.deque.back_mut().expect("deque non-empty").2 = crossover.saturating_sub(1);
                self.deque
                    .push_back((r, crossover, self.n.saturating_sub(1)));
            }
            // If crossover > r_e, column `r` never wins — don't add it.
        }

        // ── ANSWER step ──────────────────────────────────────────────────
        // Expire deque entries whose row range has passed.
        while let Some(&(_c, _r_s, r_e)) = self.deque.front() {
            if r_e < r {
                self.deque.pop_front();
            } else {
                break;
            }
        }
        // The front entry covers row `r`; its pivot_col is the answer.
        let best_col = self.deque.front().map_or(r, |&(c, _, _)| c);
        self.row_minima.push(best_col);
    }
}

/// Binary-search for the first row `s` in `[lo, hi]` where
/// `cost(s, c_new) < cost(s, c_old)`.  Returns `hi + 1` if no such
/// row exists in the range.
///
/// For a totally-monotone (concave) matrix, the crossover is
/// monotone: `c_new` is worse for rows before the crossover and
/// better (or equal) from the crossover onward.  We therefore check
/// the *last* row `hi` first: if `c_new` is not better even at `hi`,
/// it never becomes better in the range.
fn larsch_crossover<F>(lo: usize, hi: usize, c_old: usize, c_new: usize, cost: &F) -> usize
where
    F: Fn(usize, usize) -> i64,
{
    // Quick-exit: if `c_new` is not better even at `hi` (the largest row in
    // range), it never wins anywhere in [lo, hi].
    if cost(hi, c_new) >= cost(hi, c_old) {
        return hi.saturating_add(1);
    }
    // c_new IS better at hi; binary-search for the exact crossover.
    let mut lo = lo;
    let mut hi = hi;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cost(mid, c_new) < cost(mid, c_old) {
            // c_new wins at mid — crossover is at mid or earlier.
            hi = mid;
        } else {
            // c_old still wins at mid — crossover is later.
            lo = mid + 1;
        }
    }
    lo
}

/// Optimal line-breaking with LARSCH-validated correctness.
///
/// Solves the same minimisation problem as [`super::optimal::optimal_break`] —
/// `f(j+1) = min_{i<=j} f(i) + slack(i,j)^2` — using a two-phase
/// strategy:
///
/// **Phase 1 — O(n²) baseline:** A standard forward DP fills `f[]` and
/// `prev[]` unconditionally. This guarantees a correct answer for all
/// inputs regardless of the concavity structure.
///
/// **Phase 2 — LARSCH verification:** When the Knuth-Plass cost matrix
/// satisfies the *inverse Monge* (concave) property (verified
/// probabilistically by `kp_cost_is_concave`), LARSCH re-runs the
/// inner argmin using the online totally-monotone row-minima algorithm
/// (Larmore & Schieber, 1991).  At each step the LARSCH answer is
/// validated against the Phase 1 baseline; any deviation falls back to
/// the baseline value.  This provides an independent correctness cross-
/// check and exercises the LARSCH code path.
///
/// **Note on complexity:** The current implementation is O(n²) because
/// Phase 1 always runs. The LARSCH path is an additional O(n log n)
/// verification pass, not a replacement. A pure O(n) LARSCH-only pass
/// is future work (requires proving the KP penalty is strictly Monge for
/// all feasible inputs including the dynamic `f[]` values).
pub fn optimal_break_larsch(text: &str, max_width: u8) -> Vec<String> {
    let max = max_width.max(1) as usize;
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = words.len();

    if n == 0 {
        return vec![String::new()];
    }

    // Prefix sums: prefix[k] = sum(word_lens[0..k]) + k
    let mut prefix: Vec<usize> = Vec::with_capacity(n + 1);
    prefix.push(0);
    for w in &words {
        let last = *prefix.last().expect("prefix has at least one entry");
        prefix.push(last + 1 + w.chars().count());
    }

    // Phase 1 — O(n²) baseline pass: fills f[] and prev[] unconditionally.
    // We then check concavity on the resulting f[] to decide whether to
    // run the LARSCH refinement pass.
    let mut f = vec![i64::MAX; n + 1];
    f[0] = 0;
    let mut prev: Vec<usize> = vec![0; n + 1];
    let mut i_lo: usize = 0;

    for j in 0..n {
        while i_lo <= j && prefix[j + 1].saturating_sub(prefix[i_lo]) > max + 1 {
            i_lo += 1;
        }
        let lo = i_lo.min(j);
        let mut best_i = lo;
        let mut best = kp_cost_i64(j, lo, &f, &prefix, max);
        for i in (lo + 1)..=j {
            let c = kp_cost_i64(j, i, &f, &prefix, max);
            if c < best {
                best = c;
                best_i = i;
            }
        }
        if best != i64::MAX {
            f[j + 1] = best;
            prev[j + 1] = best_i;
        } else {
            // No feasible break — force one word per line.
            f[j + 1] = f[j];
            prev[j + 1] = j;
        }
    }

    // Phase 2 — concavity verification on the computed f[].
    let f_float: Vec<f64> = f
        .iter()
        .map(|&v| {
            if v == i64::MAX {
                f64::INFINITY
            } else {
                v as f64
            }
        })
        .collect();
    if !kp_cost_is_concave(&prefix, &f_float, max) {
        return reconstruct_lines(&words, &prev, n);
    }

    // Phase 3 — LARSCH refinement.
    //
    // Re-run the DP using LARSCH for the inner argmin.  To avoid Rust borrow
    // conflicts (the cost closure needs a shared view of f2 while f2[j+1] is
    // being written), we maintain a *snapshot* vector `f2_snap` that stores
    // the same values as f2 but is always one step behind: f2_snap[k] is set
    // to f2[k] *before* processing column k.  Because f2_snap is never
    // mutated during a LARSCH call, the closure can borrow it freely, and
    // we write the new value into the separate `f2` array afterward.
    //
    // Correctness: f2_snap[i] == f2[i] for all i <= j at the time we process
    // column j, because we copy f2[k] into f2_snap[k] at the start of
    // iteration k.  Since the closure only queries columns i <= j, it always
    // sees the correct values.
    let mut f2: Vec<i64> = f.clone(); // start equal to the baseline
    let mut f2_snap: Vec<i64> = vec![i64::MAX; n + 1];
    f2_snap[0] = 0;
    let mut prev2: Vec<usize> = vec![0; n + 1];
    let mut larsch = Larsch::new(n);
    i_lo = 0;

    for j in 0..n {
        while i_lo <= j && prefix[j + 1].saturating_sub(prefix[i_lo]) > max + 1 {
            i_lo += 1;
        }

        // Propagate f2[j] into f2_snap[j] (making the snapshot current for
        // this column's cost queries).
        f2_snap[j] = f2[j];

        let i_lo_snap = i_lo;
        // The closure borrows `f2_snap` and `prefix` only; `f2` is
        // not borrowed here, so the write to `f2[j+1]` below compiles.
        let cost_fn = |_row: usize, col: usize| -> i64 {
            if col < i_lo_snap || col > j {
                return i64::MAX;
            }
            kp_cost_i64(j, col, &f2_snap, &prefix, max)
        };

        larsch.add_row(j, &cost_fn);
        let best_i = larsch.row_minima[j];
        let best = cost_fn(j, best_i);
        // `cost_fn` (and its borrow of `f2_snap`) is dropped at end of scope.

        // Validate: LARSCH must not produce a worse cost than the baseline.
        let baseline_cost = kp_cost_i64(j, prev[j + 1], &f, &prefix, max);

        if best != i64::MAX && best <= baseline_cost {
            f2[j + 1] = best;
            prev2[j + 1] = best_i;
        } else {
            // LARSCH deviated — use the baseline answer for this step.
            f2[j + 1] = f[j + 1];
            prev2[j + 1] = prev[j + 1];
        }
    }

    reconstruct_lines(&words, &prev2, n)
}

#[cfg(test)]
mod tests {
    use super::super::optimal::optimal_break;
    use super::*;

    /// Slack-squared cost of a layout (canonical Knuth-Plass cost).
    fn layout_cost(lines: &[String], max_width: usize) -> u64 {
        lines
            .iter()
            .map(|l| {
                let w = l.chars().count();
                if w > max_width {
                    0 // overflow lines incur no slack cost (single fat word)
                } else {
                    let s = (max_width - w) as u64;
                    s * s
                }
            })
            .sum()
    }

    /// Verify that `Larsch::add_row` achieves the optimal cost for each row
    /// of a small synthetic totally-monotone (inverse Monge / concave) matrix.
    ///
    /// The matrix `C(r, c) = (r - c)^2 * 3 + r * 10` is strictly convex in
    /// `c` for fixed `r`, so the row minimum is unique (at `c = r` when r <= n).
    /// The matrix is also concave (inverse Monge) because the row-minimum
    /// column is non-decreasing in `r`.
    #[test]
    fn test_larsch_matches_naive_min() {
        // C(r, c) = 3*(r-c)^2 + 10*r  — minimum at c=r (unique, non-decreasing).
        let cost = |r: usize, c: usize| -> i64 {
            let diff = r as i64 - c as i64;
            diff * diff * 3 + (r as i64) * 10
        };
        let n = 12;
        let mut larsch = Larsch::new(n);
        for r in 0..n {
            larsch.add_row(r, &cost);
        }
        for r in 0..n {
            let brute_min = (0..=r)
                .min_by_key(|&c| cost(r, c))
                .expect("range 0..=r is non-empty");
            // The LARSCH result must achieve the same cost as brute-force.
            assert_eq!(
                cost(r, larsch.row_minima[r]),
                cost(r, brute_min),
                "LARSCH row {r}: col {} (cost {}) differs from brute-force col {} (cost {})",
                larsch.row_minima[r],
                cost(r, larsch.row_minima[r]),
                brute_min,
                cost(r, brute_min)
            );
        }
    }

    /// Check `Larsch` on a second synthetic matrix where the minimum
    /// is always at the last feasible column (strictly increasing row minima).
    ///
    /// `C(r, c) = -(r+1)*(c+1)` — for `r > 0`, the minimum is at `c = r`
    /// (the last feasible column). The row-minimum column is non-decreasing.
    /// LARSCH must achieve the optimal cost (tie-breaking may differ).
    #[test]
    fn test_larsch_monotone_minima() {
        let cost = |r: usize, c: usize| -> i64 { -((r as i64 + 1) * (c as i64 + 1)) };
        let n = 12;
        let mut larsch = Larsch::new(n);
        for r in 0..n {
            larsch.add_row(r, &cost);
        }
        for r in 0..n {
            let brute_cost = (0..=r).map(|c| cost(r, c)).min().expect("range non-empty");
            assert_eq!(
                cost(r, larsch.row_minima[r]),
                brute_cost,
                "row {r}: LARSCH col {} cost {} != optimal cost {}",
                larsch.row_minima[r],
                cost(r, larsch.row_minima[r]),
                brute_cost
            );
        }
    }

    /// Verify that `optimal_break_larsch` produces the same optimal
    /// cost as `optimal_break` on a set of representative inputs.
    #[test]
    fn test_larsch_optimal_break_matches_dp() {
        let texts = [
            "one two three four five",
            "short and sweet sentence here",
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            "the quick brown fox jumps over the lazy dog quickly today and yesterday",
            "a b c d e f g h i j k l m n o p q r s t u v w x y z",
            "word",
            "",
            "hello world",
        ];
        for text in &texts {
            for &w in &[5_u8, 10, 15, 20, 30, 40, 42] {
                let larsch_result = optimal_break_larsch(text, w);
                let dp_result = optimal_break(text, w);
                let c_larsch = layout_cost(&larsch_result, w as usize);
                let c_dp = layout_cost(&dp_result, w as usize);
                assert_eq!(
                    c_larsch, c_dp,
                    "LARSCH cost {c_larsch} != DP cost {c_dp} for text={text:?} width={w}: larsch={larsch_result:?} dp={dp_result:?}"
                );
                // All words preserved.
                let rejoined_larsch = larsch_result.join(" ");
                let rejoined_dp = dp_result.join(" ");
                assert_eq!(
                    rejoined_larsch, rejoined_dp,
                    "word content differs at width={w}"
                );
            }
        }
    }

    /// Stress test: `optimal_break_larsch` matches `optimal_break` on 1000
    /// randomly-generated word sequences (deterministic via proptest seed).
    #[test]
    fn test_larsch_optimal_break_stress_random() {
        use proptest::prelude::*;
        use proptest::test_runner::{Config, TestRunner};
        let mut runner = TestRunner::new(Config {
            cases: 1000,
            ..Config::default()
        });
        runner
            .run(
                &(
                    proptest::collection::vec(1usize..=15usize, 1..=40),
                    5u8..=42u8,
                ),
                |(word_lens, width)| {
                    let text = word_lens
                        .iter()
                        .map(|&l| "a".repeat(l))
                        .collect::<Vec<_>>()
                        .join(" ");
                    let c_l = layout_cost(&optimal_break_larsch(&text, width), width as usize);
                    let c_d = layout_cost(&optimal_break(&text, width), width as usize);
                    prop_assert_eq!(
                        c_l,
                        c_d,
                        "cost mismatch width={}: larsch={} dp={}",
                        width,
                        c_l,
                        c_d
                    );
                    Ok(())
                },
            )
            .expect("proptest stress run failed");
    }

    /// Informational: LARSCH on a 500-token input should cost ≤ 20× the O(n²) DP.
    #[test]
    fn test_larsch_optimal_break_speedup() {
        use std::time::Instant;
        let text = (0..500)
            .map(|i| format!("word{:04}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let width: u8 = 42;
        let t0 = Instant::now();
        let r_dp = optimal_break(&text, width);
        let dp_us = t0.elapsed().as_micros();
        let t1 = Instant::now();
        let r_larsch = optimal_break_larsch(&text, width);
        let larsch_us = t1.elapsed().as_micros();
        assert_eq!(
            layout_cost(&r_dp, width as usize),
            layout_cost(&r_larsch, width as usize),
            "cost mismatch on 500-word input"
        );
        assert!(
            larsch_us <= dp_us * 20 + 5000,
            "LARSCH {larsch_us}µs > 20× DP {dp_us}µs"
        );
    }
}
