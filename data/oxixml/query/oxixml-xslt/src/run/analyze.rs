//! `xsl:analyze-string`: splitting an input string into matching and non-matching substrings.
//!
//! [XSLT 3.0 §15](https://www.w3.org/TR/xslt-30/#analyze-string) runs
//! `xsl:matching-substring` once per match with `regex-group(n)` bound to the captured groups,
//! and `xsl:non-matching-substring` once per gap. [`oxixml_regex::Regex::find_iter`] supplies
//! the matches directly — byte ranges into `input`, with each capturing group's own range —
//! which is the right primitive for this instruction specifically because it does *not*
//! reject a pattern that can match a zero-length string; see the note on `FORX0003` below.

use super::{Context, Emitter, Transformer};
use crate::compile::{Avt, Instruction};
use crate::error::{XsltError, XsltResult};
use oxixml_regex::Regex;

/// One piece of the analysed string.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Piece {
    /// The text of the piece.
    pub(crate) text: String,
    /// The capture groups, with index 0 the whole match. Empty for a non-matching piece.
    pub(crate) groups: Vec<String>,
    /// Whether the regular expression matched here.
    pub(crate) matched: bool,
}

/// The string the `select` attribute of `xsl:analyze-string` supplies.
///
/// [XSLT 3.0 §15](https://www.w3.org/TR/xslt-30/#analyze-string) gives the attribute the
/// required type `xs:string`, so the function conversion rules govern it rather than the
/// string-value-of-a-sequence rule that governs `xsl:value-of`. A node is atomized to an untyped
/// atomic and cast, which is the ordinary case; but a sequence of more than one item is a type
/// error (`analyze-string-003` supplies `('red','green','blue')`), and so is an atomic value of
/// a type the rules do not cast to `xs:string` (`analyze-string-001` supplies the integer `22`).
///
/// An empty sequence is treated as the zero-length string, which is what the cardinality of a
/// stylesheet's own `select=""` amounts to and what every case in the suite that selects a
/// missing node relies on.
///
/// # Errors
///
/// `XPTY0004`.
fn coerce_to_string(value: &oxixml_xpath::XpathValue) -> XsltResult<String> {
    let mut out: Option<String> = None;
    for item in value.iter() {
        if out.is_some() {
            return Err(XsltError::dynamic(
                "XPTY0004",
                "the select attribute of xsl:analyze-string must supply a single string",
            ));
        }
        let text = match item.as_atomic() {
            // The function conversion rules cast an untyped atomic — which is what atomizing a
            // node gives — and accept a string or a URI; nothing else reaches `xs:string`.
            Some(atomic) if !castable_to_string(atomic) => {
                return Err(XsltError::dynamic(
                    "XPTY0004",
                    format!(
                        "the select attribute of xsl:analyze-string requires xs:string, not {}",
                        atomic.atomic_type()
                    ),
                ));
            }
            _ => item.string_value()?,
        };
        out = Some(text);
    }
    Ok(out.unwrap_or_default())
}

/// Whether the function conversion rules turn `value` into an `xs:string`.
fn castable_to_string(value: &oxixml_infoset::AtomicValue) -> bool {
    use oxixml_infoset::AtomicType;
    matches!(
        value.atomic_type(),
        AtomicType::UntypedAtomic
            | AtomicType::AnyUri
            | AtomicType::String
            | AtomicType::NormalizedString
            | AtomicType::Token
            | AtomicType::Language
            | AtomicType::Nmtoken
            | AtomicType::Name
            | AtomicType::Ncname
            | AtomicType::Id
            | AtomicType::Idref
            | AtomicType::Entity
    )
}

/// Splits `input` into alternating non-matching and matching pieces.
///
/// A pattern that can match a zero-length string — `a*`, or even the empty pattern itself — is
/// *not* an error here, unlike `fn:replace` and `fn:tokenize`: XSLT 2.0's `xsl:analyze-string`
/// rejected one with `XTDE1150`, but [XSLT 3.0 §15](https://www.w3.org/TR/xslt-30/#analyze-string)
/// deliberately relaxed that restriction (the suite's analyze-string-090a/091a, pinned to
/// `XSLT20` alone, are the withdrawn behaviour; -090b/-091b/-092, `XSLT30+`, are this one).
/// [`Regex::find_iter`] already implements the resulting rule directly: a zero-length match
/// resumes the scan one `char` further on rather than repeating forever, exactly the algorithm
/// [XSLT 3.0 §15.1](https://www.w3.org/TR/xslt-30/#func-analyze-string) itself describes.
///
/// # Errors
///
/// `FORX0002` for a malformed pattern.
pub(crate) fn analyze(input: &str, pattern: &str, flags: &str) -> XsltResult<Vec<Piece>> {
    let regex = Regex::xpath(pattern, flags).map_err(|error| {
        XsltError::dynamic("FORX0002", format!("invalid regular expression: {error}"))
    })?;
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let mut pieces = Vec::new();
    let mut last = 0;
    for found in regex.find_iter(input) {
        // A back-reference pattern can exhaust the back-tracking budget partway through the
        // input; stopping the scan there and treating the remainder as unmatched is the same
        // choice `fn:analyze-string` (`oxixml_fn::strings::analyze_string`) makes.
        let Ok(found) = found else { break };
        let range = found.range();
        if range.start > last {
            pieces.push(Piece {
                text: input[last..range.start].to_owned(),
                groups: Vec::new(),
                matched: false,
            });
        }
        let mut groups = Vec::with_capacity(found.group_count() + 1);
        groups.push(found.as_str().to_owned());
        groups.extend(found.groups().map(|group| {
            group
                .map(|range| input[range].to_owned())
                .unwrap_or_default()
        }));
        pieces.push(Piece {
            text: found.as_str().to_owned(),
            groups,
            matched: true,
        });
        last = range.end;
    }
    if last < input.len() {
        pieces.push(Piece {
            text: input[last..].to_owned(),
            groups: Vec::new(),
            matched: false,
        });
    }
    Ok(pieces)
}

impl Transformer {
    /// Runs one `xsl:analyze-string`.
    ///
    /// # Errors
    ///
    /// Any dynamic error the select expression, the regular expression or the bodies raise.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn analyze_string(
        &mut self,
        select: &oxixml_xpath::CompiledXPath,
        regex: &Avt,
        flags: &Avt,
        matching: &Option<Vec<Instruction>>,
        non_matching: &Option<Vec<Instruction>>,
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<super::exec::Flow> {
        let input = coerce_to_string(&self.evaluate(select, context)?)?;
        let pattern = self.evaluate_avt(regex, context)?;
        let flags = self.evaluate_avt(flags, context)?;
        // XSLT 3.0 §15 gives `xsl:analyze-string` its own codes for a malformed `regex` and an
        // unrecognized `flags`, in place of the `FORX` codes `fn:analyze-string` raises.
        if flags.chars().any(|flag| !"smixq".contains(flag)) {
            return Err(XsltError::dynamic(
                "XTDE1145",
                format!("`{flags}` is not a legal value for the flags attribute"),
            ));
        }
        let pieces = analyze(&input, &pattern, &flags).map_err(|error| {
            if error.is("FORX0002") {
                XsltError::dynamic("XTDE1140", error.message())
            } else {
                error
            }
        })?;
        // XSLT 3.0 §15: "In processing each substring, the contents of the substring will be
        // the context item (as a value of type xs:string); *the position of the substring
        // within the sequence of matching and non-matching substrings will be the context
        // position*; and *the number of matching and non-matching substrings will be the
        // context size*." The count spans both kinds, and skipping a substring whose element
        // is absent does not renumber the rest, so the position comes from the index in the
        // whole partition rather than from a counter over the bodies actually run.
        let size = pieces.len();
        for (offset, piece) in pieces.into_iter().enumerate() {
            let body = if piece.matched {
                matching
            } else {
                non_matching
            };
            let Some(body) = body else { continue };
            let scoped = context.with_focus(
                oxixml_infoset::Item::Atomic(oxixml_infoset::AtomicValue::string(
                    piece.text.clone(),
                )),
                offset + 1,
                size,
            );
            if piece.matched {
                self.runtime.regex_groups.borrow_mut().push(piece.groups);
            }
            let flow = self.run_sequence(body, &scoped, out);
            if piece.matched {
                self.runtime.regex_groups.borrow_mut().pop();
            }
            let flow = flow?;
            if !matches!(flow, super::exec::Flow::Normal) {
                return Ok(flow);
            }
        }
        Ok(super::exec::Flow::Normal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_and_non_matching_pieces_alternate() -> XsltResult<()> {
        let pieces = analyze("a1b22c", "[0-9]+", "")?;
        let texts: Vec<&str> = pieces.iter().map(|piece| piece.text.as_str()).collect();
        assert_eq!(texts, ["a", "1", "b", "22", "c"]);
        assert_eq!(
            pieces.iter().map(|piece| piece.matched).collect::<Vec<_>>(),
            [false, true, false, true, false]
        );
        Ok(())
    }

    #[test]
    fn capture_groups_are_reported() -> XsltResult<()> {
        let pieces = analyze("2024-06", "([0-9]{4})-([0-9]{2})", "")?;
        let matched = pieces.iter().find(|piece| piece.matched).expect("a match");
        assert_eq!(matched.groups, ["2024-06", "2024", "06"]);
        Ok(())
    }

    #[test]
    fn an_empty_input_yields_nothing() -> XsltResult<()> {
        assert!(analyze("", "a", "")?.is_empty());
        Ok(())
    }

    #[test]
    fn a_pattern_matching_the_empty_string_is_not_an_error() -> XsltResult<()> {
        // XSLT 3.0 §15 relaxed XSLT 2.0's `XTDE1150`: `a*` matches "a" outright, then a
        // zero-length match at every position that is not itself an `a`, advancing one `char`
        // at a time rather than looping forever.
        let pieces = analyze("abc", "a*", "")?;
        let texts: Vec<&str> = pieces.iter().map(|piece| piece.text.as_str()).collect();
        let matched: Vec<bool> = pieces.iter().map(|piece| piece.matched).collect();
        assert_eq!(texts, ["a", "", "b", "", "c", ""]);
        assert_eq!(matched, [true, true, false, true, false, true]);
        Ok(())
    }

    #[test]
    fn an_empty_pattern_matches_a_zero_length_string_at_every_position() -> XsltResult<()> {
        // analyze-string-091b: `regex=""` — the most extreme zero-length-only pattern there
        // is — is likewise not an error, and splits the input at every character boundary.
        let pieces = analyze("abcdef", "", "")?;
        let non_matching: Vec<&str> = pieces
            .iter()
            .filter(|piece| !piece.matched)
            .map(|piece| piece.text.as_str())
            .collect();
        assert_eq!(non_matching, ["a", "b", "c", "d", "e", "f"]);
        assert!(
            pieces
                .iter()
                .all(|piece| !piece.matched || piece.text.is_empty())
        );
        Ok(())
    }

    #[test]
    fn a_pattern_that_can_match_empty_or_non_empty_finds_both() -> XsltResult<()> {
        // analyze-string-092's shape: a CSV-style alternation whose second branch can match
        // zero characters (an empty field between two commas) while its first branch matches
        // real, non-empty quoted content — both must be usable in the same scan.
        let pieces = analyze("a,,b", "(?:^|,)([^,]*)", "")?;
        let matched: Vec<&str> = pieces
            .iter()
            .filter(|piece| piece.matched)
            .map(|piece| piece.groups[1].as_str())
            .collect();
        assert_eq!(matched, ["a", "", "b"]);
        Ok(())
    }
}
