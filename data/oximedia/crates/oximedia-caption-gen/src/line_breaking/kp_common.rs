//! Shared Knuth-Plass primitives used by the SMAWK and LARSCH breakers.

/// Compute the Knuth-Plass cost `f[i] + slack(i,j)^2` as an `i64`.
///
/// Returns `i64::MAX` for infeasible entries (line too wide, or `f[i]` not
/// yet populated).
#[inline(always)]
pub(super) fn kp_cost_i64(
    j: usize,
    i: usize,
    f: &[i64],
    prefix: &[usize],
    max_width: usize,
) -> i64 {
    if i > j {
        return i64::MAX;
    }
    let fi = f[i];
    if fi == i64::MAX {
        return i64::MAX;
    }
    let w = if prefix[j + 1] > prefix[i] + 1 {
        prefix[j + 1] - prefix[i] - 1
    } else {
        0
    };
    if w > max_width {
        return i64::MAX;
    }
    let slack = (max_width - w) as i64;
    fi.saturating_add(slack * slack)
}

/// Verify that the Knuth-Plass cost function is concave (inverse Monge)
/// for the given prefix-sum array and DP vector on a small random sample.
///
/// The check is probabilistic: it samples `check_pairs` pairs of
/// `(i, i', j)` with `i < i' <= j` and verifies the inverse Monge
/// condition `C(i',j) + C(i,j') >= C(i,j) + C(i',j')` for `j' = j+1`.
/// Returns `true` if all sampled pairs satisfy the condition.
pub(super) fn kp_cost_is_concave(prefix: &[usize], f: &[f64], max_width: usize) -> bool {
    let n = f.len().saturating_sub(1);
    if n < 4 {
        return true;
    }
    // Check a small set of triples: (i, i', j) where i < i' and j >= i'.
    // We verify the Monge condition: C(i,j) + C(i',j+1) <= C(i,j+1) + C(i',j).
    let sample_limit = 64.min(n * (n - 1) / 2);
    let mut checked = 0usize;
    let step = (n * (n - 1) / 2 / sample_limit).max(1);

    // Inline cost closure (mirrors KpForwardMatrix::cost but in integer arithmetic)
    let cost = |row: usize, col: usize| -> f64 {
        if row > col || col >= n {
            return f64::INFINITY;
        }
        let pv = f[row];
        if !pv.is_finite() {
            return f64::INFINITY;
        }
        // line width = prefix[col+1] - prefix[row] - 1
        let w = prefix[col + 1]
            .saturating_sub(prefix[row])
            .saturating_sub(1);
        if w > max_width {
            return f64::INFINITY;
        }
        let slack = (max_width - w) as f64;
        pv + slack * slack
    };

    let mut pair_idx = 0usize;
    'outer: for i in 0..n {
        for ip in (i + 1)..n {
            if pair_idx % step != 0 {
                pair_idx += 1;
                continue;
            }
            pair_idx += 1;
            // j must be >= ip and j+1 < n
            let j = ip;
            if j + 1 >= n {
                continue;
            }
            let cij = cost(i, j);
            let cipjp = cost(ip, j + 1);
            let cijp = cost(i, j + 1);
            let cipj = cost(ip, j);
            // All finite: check inverse Monge
            if cij.is_finite() && cipjp.is_finite() && cijp.is_finite() && cipj.is_finite() {
                // Inverse Monge: C(i,j) + C(i',j') >= C(i,j') + C(i',j)
                if (cij + cipjp) < (cijp + cipj) - 1e-9 {
                    return false;
                }
                checked += 1;
                if checked >= sample_limit {
                    break 'outer;
                }
            }
        }
    }
    true
}

/// Reconstruct lines from the `prev` backtrace array.
pub(super) fn reconstruct_lines(words: &[&str], prev: &[usize], n: usize) -> Vec<String> {
    let mut starts: Vec<usize> = Vec::new();
    let mut k = n;
    let mut guard = 0usize;
    while k > 0 {
        let p = prev[k];
        let next_k = if p == k { k.saturating_sub(1) } else { p };
        starts.push(next_k);
        guard += 1;
        if guard > n + 2 {
            break;
        }
        k = next_k;
    }
    starts.reverse();
    starts.push(n);

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
