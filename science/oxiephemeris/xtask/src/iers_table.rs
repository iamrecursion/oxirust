//! Parser for the IERS Conventions (2010) electronic nutation tables
//! `tab5.3a.txt` (nutation in longitude) and `tab5.3b.txt` (nutation in
//! obliquity), IAU `2000A_R06` expression.
//!
//! Format (see the file headers): after a `j = 0  Number of terms = N`
//! separator come `N` data rows of 17 whitespace-separated fields — a
//! 1-based index, two amplitudes in microarcseconds, and the 14 integer
//! multipliers of the fundamental arguments `l l' F D Om L_Me L_Ve L_E
//! L_Ma L_J L_Sa L_U L_Ne p_A` (IERS TN36 eqs. 5.43/5.44); then the same
//! for `j = 1` (amplitudes in µas per Julian century). In *both* files
//! the first amplitude column multiplies `sin(ARG)` and the second
//! multiplies `cos(ARG)` (tab5.3a prints `A_i  A''_i`; tab5.3b prints
//! `B''_i  B_i` — note the swapped header order there).
//!
//! Amplitudes are kept as their verbatim decimal strings so the code
//! generator can reproduce them digit-for-digit.

use std::path::Path;

/// One data row: 14 fundamental-argument multipliers plus the verbatim
/// `sin`/`cos` amplitude strings (µas or µas/cy).
pub(crate) struct Term {
    /// Multipliers of `l, l', F, D, Om, L_Me, L_Ve, L_E, L_Ma, L_J, L_Sa,
    /// L_U, L_Ne, p_A`, in table column order.
    pub(crate) mult: [i8; 14],
    /// Amplitude of `sin(ARG)`, verbatim decimal string from the table.
    pub(crate) sin_amp: String,
    /// Amplitude of `cos(ARG)`, verbatim decimal string from the table.
    pub(crate) cos_amp: String,
}

impl Term {
    /// `true` when every planetary multiplier (`L_Me..p_A`) vanishes,
    /// i.e. the argument involves only the five Delaunay variables.
    pub(crate) fn is_luni_solar(&self) -> bool {
        self.mult[5..].iter().all(|&m| m == 0)
    }
}

/// A fully parsed table: the `j = 0` (constant) and `j = 1` (t-linear)
/// blocks, validated against the term counts declared in the block
/// headers.
pub(crate) struct Table {
    /// `j = 0` terms (amplitudes in µas).
    pub(crate) j0: Vec<Term>,
    /// `j = 1` terms (amplitudes in µas per Julian century TT).
    pub(crate) j1: Vec<Term>,
}

/// Checks that an amplitude field looks like `-?digits.digits` so that it
/// can be embedded verbatim as a Rust `f64` literal.
fn validate_amplitude(raw: &str) -> Result<String, String> {
    let unsigned = raw.strip_prefix('-').unwrap_or(raw);
    let well_formed = unsigned
        .split_once('.')
        .is_some_and(|(int_part, frac_part)| {
            !int_part.is_empty()
                && !frac_part.is_empty()
                && int_part.bytes().all(|b| b.is_ascii_digit())
                && frac_part.bytes().all(|b| b.is_ascii_digit())
        });
    if well_formed {
        Ok(raw.to_owned())
    } else {
        Err(format!("unexpected amplitude format {raw:?}"))
    }
}

/// Parses one table file and validates the block term counts.
pub(crate) fn parse_table(path: &Path) -> Result<Table, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    // (declared term count, parsed terms) per `j = ...` block.
    let mut blocks: Vec<(usize, Vec<Term>)> = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let lineno = idx + 1;
        let at = |msg: String| format!("{}:{lineno}: {msg}", path.display());
        let trimmed = line.trim();
        if trimmed.starts_with("j = ") {
            // e.g. `j = 0  Number of terms = 1320`
            let declared = trimmed
                .rsplit('=')
                .next()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .ok_or_else(|| at(format!("unparsable block header {trimmed:?}")))?;
            blocks.push((declared, Vec::new()));
            continue;
        }
        let Some((_, terms)) = blocks.last_mut() else {
            continue; // preamble before the first block
        };
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() != 17 || fields[0].parse::<u32>().is_err() {
            continue; // block header / separator / column-title lines
        }
        let sin_amp = validate_amplitude(fields[1]).map_err(&at)?;
        let cos_amp = validate_amplitude(fields[2]).map_err(&at)?;
        let mut mult = [0_i8; 14];
        for (slot, raw) in mult.iter_mut().zip(&fields[3..17]) {
            *slot = raw
                .parse::<i32>()
                .ok()
                .and_then(|v| i8::try_from(v).ok())
                .ok_or_else(|| at(format!("bad multiplier {raw:?}")))?;
        }
        terms.push(Term {
            mult,
            sin_amp,
            cos_amp,
        });
    }
    let mut it = blocks.into_iter();
    match (it.next(), it.next(), it.next()) {
        (Some((n0, j0)), Some((n1, j1)), None) => {
            if j0.len() != n0 || j1.len() != n1 {
                return Err(format!(
                    "{}: parsed {}/{} terms but headers declare {n0}/{n1}",
                    path.display(),
                    j0.len(),
                    j1.len()
                ));
            }
            Ok(Table { j0, j1 })
        }
        _ => Err(format!(
            "{}: expected exactly two `j = ...` blocks",
            path.display()
        )),
    }
}
