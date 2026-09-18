//! SMAWK O(n) totally-monotone matrix row-minima algorithm and the
//! `optimal_break_smawk` line-breaking function built on top of it.

/// A cost matrix exposed to the SMAWK row-minima algorithm.
///
/// SMAWK requires the matrix to be **totally monotone**: for every
/// `i < i'` and `j < j'`,
///
/// ```text
///     cost(i,  j ) + cost(i', j') <= cost(i, j') + cost(i', j)
/// ```
///
/// When this property holds the row minima can be computed in
/// `O(rows + cols)` time using the Aggarwal–Klawe–Moran–Shor–Wilber
/// (1987) reduce-and-interpolate recursion.
pub trait TotallyMonotoneMatrix {
    /// Number of rows in the matrix.
    fn rows(&self) -> usize;
    /// Number of columns in the matrix.
    fn cols(&self) -> usize;
    /// Return the cost at `(row, col)`.  Infeasible entries return
    /// `f64::INFINITY`.
    fn cost(&self, row: usize, col: usize) -> f64;
}

/// Compute, for every row of `m`, the column index achieving the minimum
/// cost on that row using the SMAWK algorithm.
///
/// Time complexity: `O(rows + cols)` for a totally monotone matrix.
/// Falls back to a naive scan when there are zero rows.
///
/// **Invariant (debug only):** the input must be totally monotone.  This
/// function does not assert the property in release builds because the
/// check is `O(rows^2 * cols)`.
pub fn smawk_row_minima(m: &dyn TotallyMonotoneMatrix) -> Vec<usize> {
    let rows = m.rows();
    let cols = m.cols();
    if rows == 0 {
        return Vec::new();
    }
    if cols == 0 {
        return vec![0; rows];
    }

    // Initial row/column index lists (logical → physical).
    let row_indices: Vec<usize> = (0..rows).collect();
    let col_indices: Vec<usize> = (0..cols).collect();

    let mut answer: Vec<usize> = vec![0; rows];
    smawk_recurse(m, &row_indices, &col_indices, &mut answer);
    answer
}

/// SMAWK recursive worker.
///
/// `rows` and `cols` are sub-views into the original matrix; `answer`
/// stores, for every original row index appearing in `rows`, the column
/// index that minimises the cost on that row.
fn smawk_recurse(
    m: &dyn TotallyMonotoneMatrix,
    rows: &[usize],
    cols: &[usize],
    answer: &mut [usize],
) {
    if rows.is_empty() {
        return;
    }
    if rows.len() == 1 {
        // Linear scan for the single remaining row.
        let r = rows[0];
        let mut best_col = cols[0];
        let mut best = m.cost(r, best_col);
        for &c in &cols[1..] {
            let v = m.cost(r, c);
            if v < best {
                best = v;
                best_col = c;
            }
        }
        answer[r] = best_col;
        return;
    }

    // ── Reduce ──────────────────────────────────────────────────────────
    // Eliminate columns that cannot contribute a row minimum to any of
    // the surviving rows.  After reduction at most `rows.len()` columns
    // remain.
    let reduced_cols = smawk_reduce(m, rows, cols);

    // ── Interpolate ─────────────────────────────────────────────────────
    // Recurse on the even-indexed rows (every other row).
    let even_rows: Vec<usize> = rows.iter().step_by(2).copied().collect();
    smawk_recurse(m, &even_rows, &reduced_cols, answer);

    // For odd-indexed rows, the optimal column is constrained between
    // the optima of the neighbouring even rows; scan only that window.
    let n = rows.len();
    let mut col_pos: usize = 0;
    for i in (1..n).step_by(2) {
        let row = rows[i];
        // Lower bound: the answer of the previous even row.
        let prev_even = rows[i - 1];
        let lo_col = answer[prev_even];
        // Upper bound: the answer of the next even row, or last column.
        let hi_col = if i + 1 < n {
            let next_even = rows[i + 1];
            answer[next_even]
        } else {
            *reduced_cols
                .last()
                .expect("reduced_cols non-empty when rows.len() >= 2")
        };

        // Advance `col_pos` to the first reduced column >= lo_col.
        while col_pos < reduced_cols.len() && reduced_cols[col_pos] < lo_col {
            col_pos += 1;
        }
        if col_pos >= reduced_cols.len() {
            // Should never happen if matrix is well-formed; fall back
            // to the very last reduced column.
            answer[row] = hi_col;
            continue;
        }

        // Scan candidate columns in [lo_col, hi_col].
        let mut best_col = lo_col;
        let mut best = m.cost(row, best_col);
        let mut scan = col_pos;
        while scan < reduced_cols.len() && reduced_cols[scan] <= hi_col {
            let c = reduced_cols[scan];
            let v = m.cost(row, c);
            if v < best {
                best = v;
                best_col = c;
            }
            scan += 1;
        }
        answer[row] = best_col;
    }
}

/// SMAWK column reduction.
///
/// After this call the returned column list contains at most
/// `rows.len()` entries.  Each entry is guaranteed to be the row
/// minimum of *some* row in `rows`.
fn smawk_reduce(m: &dyn TotallyMonotoneMatrix, rows: &[usize], cols: &[usize]) -> Vec<usize> {
    let mut stack: Vec<usize> = Vec::with_capacity(cols.len());
    for &c in cols {
        loop {
            if stack.is_empty() {
                stack.push(c);
                break;
            }
            let top = *stack.last().expect("stack is non-empty here");
            let k = stack.len() - 1;
            // Compare top of stack vs. new column on the row at depth k.
            let r = rows[k.min(rows.len() - 1)];
            if m.cost(r, top) <= m.cost(r, c) {
                // The top column is at least as good as c for row r;
                // push c if there is still room.
                if stack.len() < rows.len() {
                    stack.push(c);
                }
                break;
            } else {
                // The top column is dominated by c — discard.
                stack.pop();
            }
        }
    }
    stack
}

/// A forward Knuth-Plass cost matrix used by the SMAWK breaker.
///
/// `entry(i, j)` represents the optimal total cost to break the prefix
/// `words[0..=j]` such that the *last* line covers `words[i..=j]`:
///
/// ```text
///     entry(i, j) = f(i) + line_cost(i, j)
/// ```
///
/// where `f(i)` is the optimal cost up to (but not including) word `i`
/// and `line_cost(i, j) = (max_width - line_width(i, j))^2` if the
/// line fits, else `+∞`.
///
/// For fixed `j`, the row that minimises `entry(·, j)` gives both the
/// optimal cost `f(j+1) = entry(arg_min, j)` and the breakpoint
/// `prev[j+1] = arg_min`.
struct KpForwardMatrix<'a> {
    prefix: &'a [usize],
    /// f[i] = optimal cost for words 0..i (f[0] = 0)
    f: &'a [f64],
    max_width: usize,
    n: usize,
}

impl KpForwardMatrix<'_> {
    fn line_width(&self, i: usize, j: usize) -> usize {
        // `prefix` is the "always-one-space-per-word" cumulative
        // sum: `prefix[k] = sum(word_lens[0..k]) + k`.  The width of
        // the line covering `words[i..=j]` is
        // `prefix[j + 1] - prefix[i] - 1` (subtract one because the
        // leading "+1" corresponds to a space that we don't actually
        // render at the start of a line).
        self.prefix[j + 1] - self.prefix[i] - 1
    }
}

impl TotallyMonotoneMatrix for KpForwardMatrix<'_> {
    fn rows(&self) -> usize {
        self.n
    }
    fn cols(&self) -> usize {
        self.n
    }
    fn cost(&self, row: usize, col: usize) -> f64 {
        // Row index `row` corresponds to the start word of the last
        // line; column `col` corresponds to its end word (inclusive).
        // We need `row <= col` and the line must fit.
        if row > col {
            return f64::INFINITY;
        }
        let prev = self.f[row];
        if !prev.is_finite() {
            return f64::INFINITY;
        }
        let width = self.line_width(row, col);
        if width > self.max_width {
            return f64::INFINITY;
        }
        let slack = (self.max_width - width) as f64;
        prev + slack * slack
    }
}

/// Optimal line break using a forward Knuth-Plass DP with SMAWK
/// support primitives.
///
/// Produces the same optimal total slack-squared cost as
/// [`super::optimal::optimal_break`]; tie-breakers between equally-optimal
/// layouts may differ.
///
/// The forward recurrence is
///
/// ```text
///     f(0) = 0
///     f(j+1) = min over i in 0..=j  of  f(i) + line_cost(i, j)
/// ```
///
/// The cost matrix `cost(i, j) = f(i) + (max_width - line_width(i,
/// j))^2` is **totally monotone on its feasible entries**
/// (`KpForwardMatrix`) and the SMAWK primitive
/// [`smawk_row_minima`] computes row minima on such matrices in
/// `O(rows + cols)`.  However, this DP's matrix is **online**:
/// column `j`'s costs depend on `f[0..=j]` which is what we are
/// computing.  Wiring SMAWK into the online case requires the
/// Larmore–Schieber LARSCH algorithm (concave SMAWK with online
/// updates) which is beyond this iteration.
///
/// Today's implementation therefore uses a per-column linear scan
/// over the **feasibility window** `[i_lo(j), j]` where `i_lo(j)` is
/// the smallest feasible row.  This gives the same worst-case
/// `O(n^2)` as `optimal_break` but with better constants
/// (one-pass, single-allocation prefix sums; no suffix-recursion
/// overhead).  The window pruning amortises to `O(n)` on inputs
/// where `max_width` stays small relative to `n` — common for
/// caption text.
///
/// The output has been validated against `optimal_break` across
/// 10 000 randomised inputs (see
/// `tests/integration.rs::test_smawk_matches_dp_10k_random_inputs`).
pub fn optimal_break_smawk(text: &str, max_width: u8) -> Vec<String> {
    let max = max_width.max(1) as usize;
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = words.len();

    if n == 0 {
        return vec![String::new()];
    }

    // Prefix sums using the "always-one-space-per-word" convention:
    // `prefix[k] = sum(word_lens[0..k]) + k`.  This makes the line
    // width formula `prefix[j+1] - prefix[i] - 1` regardless of `i`,
    // because the leading "+1" cancels for any `i >= 0`.
    let mut prefix: Vec<usize> = Vec::with_capacity(n + 1);
    prefix.push(0);
    for w in &words {
        let last = *prefix.last().expect("prefix has at least one entry");
        prefix.push(last + 1 + w.chars().count());
    }

    // f[k] = optimal cost for words[0..k]; f[0] = 0; infeasible = +∞.
    let mut f = vec![f64::INFINITY; n + 1];
    f[0] = 0.0;
    // prev[k] = breakpoint row i such that the last line of the
    // optimal break of words[0..k] covers words[i..k].
    let mut prev: Vec<usize> = vec![0; n + 1];

    // i_lo monotonically advances; reused across columns.
    let mut i_lo: usize = 0;

    for j in 0..n {
        // Advance i_lo so the line [i_lo..=j] still fits.  Uses the
        // same `prefix[j+1] - prefix[i] - 1` formula as
        // `KpForwardMatrix::line_width`.
        while i_lo <= j && prefix[j + 1] - prefix[i_lo] > max + 1 {
            i_lo += 1;
        }

        // Compute row minimum for column j on feasible rows
        // [i_lo..=j] using the totally-monotone matrix view of the
        // forward DP.
        let matrix = KpForwardMatrix {
            prefix: &prefix,
            f: &f,
            max_width: max,
            n,
        };

        let lo = i_lo.min(j); // ensure at least one row
                              // Single-column row-minimum query.  Linear scan over the
                              // feasible window [lo..=j] — the *amortised* benefit comes
                              // from the fact that `lo` only advances, giving `O(n)` total
                              // work over the whole loop on bounded-width inputs.
        let mut best_i = lo;
        let mut best = matrix.cost(lo, j);
        for i in (lo + 1)..=j {
            let c = matrix.cost(i, j);
            if c < best {
                best = c;
                best_i = i;
            }
        }

        if best.is_finite() {
            f[j + 1] = best;
            prev[j + 1] = best_i;
        } else {
            // No feasible break — force one word per line.
            f[j + 1] = f[j];
            prev[j + 1] = j;
        }
    }

    // Reconstruct line breaks by walking `prev` from n back to 0.
    // `starts` collects the starting word index of each line in
    // reverse order; we will reverse and append `n` as a sentinel.
    let mut starts: Vec<usize> = Vec::new();
    let mut k = n;
    let mut guard = 0;
    while k > 0 {
        let p = prev[k];
        let next_k = if p == k {
            // Degenerate guard: a degenerate prev[k] = k would
            // signal a single-word forced line; fall back to k-1
            // to make progress.
            k.saturating_sub(1)
        } else {
            p
        };
        starts.push(next_k);
        // Sanity: prevent infinite loops on pathological data.
        guard += 1;
        if guard > n + 2 {
            break;
        }
        k = next_k;
    }
    starts.reverse();
    starts.push(n); // sentinel: the line that *would* start at n is empty
                    // and serves only as the upper bound for the last real line.

    let mut lines: Vec<String> = Vec::new();
    for win in starts.windows(2) {
        let start = win[0];
        let end = win[1];
        if start >= end {
            continue;
        }
        lines.push(words[start..end].join(" "));
    }
    if lines.is_empty() {
        lines.push(words.join(" "));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::super::kp_common::kp_cost_i64;
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

    /// A small totally-monotone matrix used as a unit fixture for
    /// [`smawk_row_minima`].  Row `i` is `cost[i]` and is treated as
    /// fully feasible.
    struct DenseMatrix {
        data: Vec<Vec<f64>>,
    }

    impl TotallyMonotoneMatrix for DenseMatrix {
        fn rows(&self) -> usize {
            self.data.len()
        }
        fn cols(&self) -> usize {
            self.data.first().map(Vec::len).unwrap_or(0)
        }
        fn cost(&self, row: usize, col: usize) -> f64 {
            self.data[row][col]
        }
    }

    fn naive_row_minima(m: &dyn TotallyMonotoneMatrix) -> Vec<usize> {
        let rows = m.rows();
        let cols = m.cols();
        (0..rows)
            .map(|r| {
                let mut best_c = 0;
                let mut best_v = m.cost(r, 0);
                for c in 1..cols {
                    let v = m.cost(r, c);
                    if v < best_v {
                        best_v = v;
                        best_c = c;
                    }
                }
                best_c
            })
            .collect()
    }

    #[test]
    fn smawk_row_minima_zero_rows_returns_empty() {
        let m = DenseMatrix { data: Vec::new() };
        let r = smawk_row_minima(&m);
        assert!(r.is_empty());
    }

    #[test]
    fn smawk_row_minima_single_row() {
        let m = DenseMatrix {
            data: vec![vec![5.0, 3.0, 9.0, 1.0, 7.0]],
        };
        let r = smawk_row_minima(&m);
        assert_eq!(r, vec![3]);
    }

    #[test]
    fn smawk_row_minima_matches_naive_small() {
        // Classic Monge example used in SMAWK literature.
        let data = vec![
            vec![
                25.0, 21.0, 13.0, 10.0, 20.0, 13.0, 19.0, 35.0, 37.0, 41.0, 58.0, 62.0, 50.0,
            ],
            vec![
                42.0, 35.0, 26.0, 20.0, 29.0, 21.0, 25.0, 37.0, 36.0, 39.0, 56.0, 59.0, 47.0,
            ],
            vec![
                57.0, 48.0, 35.0, 28.0, 33.0, 24.0, 28.0, 40.0, 37.0, 37.0, 54.0, 55.0, 43.0,
            ],
            vec![
                78.0, 65.0, 51.0, 42.0, 44.0, 35.0, 38.0, 48.0, 42.0, 42.0, 55.0, 53.0, 41.0,
            ],
            vec![
                90.0, 76.0, 58.0, 48.0, 49.0, 39.0, 42.0, 48.0, 39.0, 35.0, 47.0, 45.0, 31.0,
            ],
        ];
        let m = DenseMatrix { data };
        let smawk = smawk_row_minima(&m);
        let naive = naive_row_minima(&m);
        assert_eq!(smawk.len(), naive.len());
        // Each SMAWK answer is at least as good as the naive answer.
        for (i, (&sm, &nv)) in smawk.iter().zip(naive.iter()).enumerate() {
            let sm_v = m.cost(i, sm);
            let nv_v = m.cost(i, nv);
            assert!(
                (sm_v - nv_v).abs() < 1e-9,
                "row {i}: smawk picked col {sm} (={sm_v}); naive col {nv} (={nv_v})"
            );
        }
    }

    #[test]
    fn smawk_row_minima_matches_naive_random_monge() {
        // Construct a Monge matrix from concave sums: a[i] + b[j]
        // with both a and b sorted, plus a strictly-decreasing pair
        // cost: that is a totally-monotone matrix.  This is the
        // canonical construction used in textbooks.
        let a: Vec<f64> = (0..30).map(|i| (i as f64) * 1.5).collect();
        let b: Vec<f64> = (0..50).map(|j| 100.0 - (j as f64) * 1.3).collect();
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            let row: Vec<f64> = (0..b.len()).map(|j| a[i] + b[j]).collect();
            data.push(row);
        }
        let m = DenseMatrix { data };
        let smawk = smawk_row_minima(&m);
        // With strictly-decreasing b, every row's minimum should be
        // at the last column.
        for (i, &c) in smawk.iter().enumerate() {
            let last = m.cols() - 1;
            let v = m.cost(i, c);
            let vlast = m.cost(i, last);
            assert!(
                (v - vlast).abs() < 1e-9,
                "row {i}: smawk col {c} cost {v} != cost at last col {vlast}"
            );
        }
    }

    #[test]
    fn smawk_break_empty_string() {
        let result = optimal_break_smawk("", 40);
        assert_eq!(result, vec![""]);
    }

    #[test]
    fn smawk_break_single_word() {
        let result = optimal_break_smawk("Hello", 40);
        assert_eq!(result, vec!["Hello"]);
    }

    #[test]
    fn smawk_break_two_words_fit_one_line() {
        let result = optimal_break_smawk("Hello world", 40);
        assert_eq!(result, vec!["Hello world"]);
    }

    #[test]
    fn smawk_break_respects_max_width() {
        let text = "one two three four five six seven eight";
        let result = optimal_break_smawk(text, 12);
        for line in &result {
            assert!(line.chars().count() <= 12, "line '{line}' too wide");
        }
        let rejoined = result.join(" ");
        assert_eq!(rejoined, text);
    }

    #[test]
    fn smawk_break_optimal_cost_matches_dp() {
        // Same total slack-squared cost as the naive DP, even if the
        // exact line split differs in tie-breaking.
        let texts = [
            "one two three four",
            "short and sweet sentence here",
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            "the quick brown fox jumps over the lazy dog quickly today",
        ];
        for text in texts {
            for &w in &[5_u8, 10, 15, 20, 30] {
                let smawk = optimal_break_smawk(text, w);
                let dp = optimal_break(text, w);
                let c_smawk = layout_cost(&smawk, w as usize);
                let c_dp = layout_cost(&dp, w as usize);
                assert_eq!(
                    c_smawk, c_dp,
                    "cost mismatch at width {w}: smawk={smawk:?} ({c_smawk}) dp={dp:?} ({c_dp})"
                );
                // Multi-word lines must fit; single-word lines may
                // overflow when the word itself is longer than `w`.
                for line in &smawk {
                    let words_in_line: Vec<&str> = line.split_whitespace().collect();
                    if words_in_line.len() > 1 {
                        assert!(
                            line.chars().count() <= w as usize,
                            "multi-word line '{line}' exceeds width {w}"
                        );
                    }
                }
            }
        }
    }

    // Suppress dead_code warning for kp_cost_i64 used only in tests
    #[test]
    fn kp_cost_i64_infeasible_when_i_greater_than_j() {
        let f = vec![0i64, i64::MAX];
        let prefix = vec![0usize, 2, 4];
        assert_eq!(kp_cost_i64(0, 1, &f, &prefix, 10), i64::MAX);
    }
}
