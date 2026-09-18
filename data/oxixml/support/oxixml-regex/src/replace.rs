//! Replacement-string handling for `fn:replace`.
//!
//! The replacement grammar and the `$N` group-numbering rule are those of
//! [*XPath and XQuery Functions and Operators 3.1*][fo] §5.6.4:
//!
//! * `$0` is the whole match; `$N` (`1 <= N <= S`) is capturing group `N`,
//!   empty if the group did not participate; `$N` with `S < N <= 9` is empty.
//! * When `N > S` and `N > 9`, the last digit is peeled off as a literal and the
//!   rule is re-applied, so `$23` with five groups yields group 2 followed by a
//!   literal `3`.
//! * A `$` must be followed by a digit unless escaped as `\$`; a `\` must be
//!   part of a `\\` or `\$` pair. Violations raise [`RegexError::InvalidReplacement`]
//!   (`FORX0004`).
//!
//! [fo]: https://www.w3.org/TR/xpath-functions-31/#func-replace

use crate::error::RegexError;

/// Validates a replacement string independently of any match, raising
/// [`RegexError::InvalidReplacement`] on a stray `$` or `\`.
pub(crate) fn validate_replacement(replacement: &str) -> Result<(), RegexError> {
    let chars: Vec<char> = replacement.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' => match chars.get(i + 1) {
                Some('\\') | Some('$') => i += 2,
                _ => {
                    return Err(RegexError::InvalidReplacement(
                        "a backslash must be part of a \\\\ or \\$ pair".to_owned(),
                    ));
                }
            },
            '$' => match chars.get(i + 1) {
                Some(c) if c.is_ascii_digit() => i += 1,
                _ => {
                    return Err(RegexError::InvalidReplacement(
                        "a dollar sign must be followed by a digit".to_owned(),
                    ));
                }
            },
            _ => i += 1,
        }
    }
    Ok(())
}

/// Expands `replacement` for one match, appending the result to `out`.
///
/// `caps` holds the capture slots of the match (slot 0/1 is the whole match);
/// `group_count` is `S`, the number of capturing groups.
pub(crate) fn expand_into(
    out: &mut String,
    replacement: &str,
    input: &[char],
    caps: &[Option<usize>],
    group_count: usize,
) {
    let chars: Vec<char> = replacement.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                // validate_replacement has already guaranteed a valid pair.
                if let Some(next) = chars.get(i + 1) {
                    out.push(*next);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            '$' => {
                let digits_start = i + 1;
                let mut digits_end = digits_start;
                while chars.get(digits_end).is_some_and(|c| c.is_ascii_digit()) {
                    digits_end += 1;
                }
                // validate_replacement guarantees at least one digit here.
                let (group, consumed) =
                    resolve_group(&chars[digits_start..digits_end], group_count);
                push_group(out, input, caps, group, group_count);
                // Any digits not used by the group number are literal text.
                for c in &chars[digits_start + consumed..digits_end] {
                    out.push(*c);
                }
                i = digits_end;
            }
            other => {
                out.push(other);
                i += 1;
            }
        }
    }
}

/// Determines the group number referenced by a run of digits and how many of
/// those digits were consumed to form it (the remainder is literal).
fn resolve_group(digits: &[char], group_count: usize) -> (usize, usize) {
    let mut end = digits.len();
    loop {
        let n = parse_saturating(&digits[..end]);
        if n <= 9 || n <= group_count || end <= 1 {
            return (n, end);
        }
        end -= 1;
    }
}

/// Parses a run of digits into a `usize`, saturating so an enormous run cannot
/// overflow.
fn parse_saturating(digits: &[char]) -> usize {
    let mut value: usize = 0;
    for c in digits {
        let d = c.to_digit(10).unwrap_or(0) as usize;
        value = value.saturating_mul(10).saturating_add(d);
    }
    value
}

/// Appends the text of group `n` (empty when out of range or non-participating).
fn push_group(
    out: &mut String,
    input: &[char],
    caps: &[Option<usize>],
    n: usize,
    group_count: usize,
) {
    if n > group_count {
        // S < n <= 9: the reference is replaced by the empty string.
        return;
    }
    let start = caps.get(2 * n).copied().flatten();
    let end = caps.get(2 * n + 1).copied().flatten();
    if let (Some(start), Some(end)) = (start, end) {
        for c in &input[start..end] {
            out.push(*c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digits(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn greedy_digit_two_digit_group() {
        // 20 groups: "$15" is group 15, both digits used.
        assert_eq!(resolve_group(&digits("15"), 20), (15, 2));
    }

    #[test]
    fn greedy_digit_peels_last_when_out_of_range() {
        // 5 groups: "$23" -> group 2 plus a literal '3'.
        assert_eq!(resolve_group(&digits("23"), 5), (2, 1));
        // 2 groups: "$15" -> group 1 plus literal '5'.
        assert_eq!(resolve_group(&digits("15"), 2), (1, 1));
    }

    #[test]
    fn greedy_digit_single_digit_over_range() {
        // 0 groups: "$1" -> N = 1 (<= 9) resolves directly to group 1.
        assert_eq!(resolve_group(&digits("1"), 0), (1, 1));
    }

    #[test]
    fn validate_accepts_and_rejects() {
        assert!(validate_replacement("$0-$1 \\$ \\\\").is_ok());
        assert!(validate_replacement("$x").is_err());
        assert!(validate_replacement("trailing$").is_err());
        assert!(validate_replacement("\\x").is_err());
        assert!(validate_replacement("ends\\").is_err());
    }
}
