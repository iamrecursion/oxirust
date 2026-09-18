//! [`fn:analyze-string`](https://www.w3.org/TR/xpath-functions-31/#func-analyze-string),
//! F&O 3.1 §5.6.5.
//!
//! The function splits its input into the substrings a regular expression matches and the
//! substrings between them, and reports the split as an XML element:
//!
//! ```text
//! <fn:analyze-string-result>
//!   <fn:non-match>b</fn:non-match>
//!   <fn:match>a<fn:group nr="1">n</fn:group></fn:match>
//!   …
//! </fn:analyze-string-result>
//! ```
//!
//! # How the `fn:group` nesting is derived
//!
//! Inside an `fn:match` the captured groups appear **nested exactly as the parentheses nest in
//! the pattern**, with the text they did not capture interleaved. The per-match group *ranges*
//! that [`oxixml_regex::Match`] reports are not enough on their own to recover that shape:
//! `(b)(x?)` and `(b(x?))` both report group 1 over `b` and an empty group 2 immediately after
//! it, yet §5.6.5 requires `<group nr="1">b</group><group nr="2"/>` for the first and
//! `<group nr="1">b<group nr="2"/></group>` for the second (suite cases `analyzeString-017`
//! and `-017a`). The nesting is therefore read *statically*, from the pattern's own parenthesis
//! structure, by `group_forest`; the per-match ranges then only say **where** each group's
//! content sits inside the match.
//!
//! A group that did not participate in a particular match contributes nothing, and a group that
//! participated repeatedly (`(an)*`) reports its last occurrence, which is what the data model
//! can express.

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{ExpandedName, NodeRef, TreeBuilder, XdmError};
use oxixml_regex::Regex;
use std::ops::Range;

/// The namespace of the result vocabulary — the standard function namespace.
const FN_NAMESPACE: &str = "http://www.w3.org/2005/xpath-functions";

/// The prefix the result element declares for [`FN_NAMESPACE`].
///
/// §5.6.5 does not prescribe a prefix, but it does require the result element and its
/// descendants to be in the function namespace, and case `analyzeString-028` requires the
/// element's in-scope namespaces to be exactly that namespace plus the implicit `xml` one.
const FN_PREFIX: &str = "fn";

/// One capturing group of the pattern, with the groups nested directly inside it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GroupNode {
    /// The group's 1-based number, in order of its opening parenthesis.
    number: usize,
    /// The groups opened (and closed) inside this one.
    children: Vec<GroupNode>,
}

/// Reads the capturing-group nesting of an XPath-dialect pattern.
///
/// Only the parenthesis structure matters, so the scan needs to recognise exactly three things
/// that make a `(` not a group opener: a backslash escape, a character-class body (where `(`
/// and `)` are literal), and the `(?:` non-capturing form. Character classes nest, because the
/// XSD dialect spells subtraction as `[a-z-[aeiou]]`.
///
/// The `q` flag makes the whole pattern a literal, so it declares no groups at all; the `x`
/// flag removes whitespace outside character classes before the pattern is parsed, which can
/// join a `(` to a following `?:`, so it is applied here too.
fn group_forest(pattern: &str, flags: &str) -> Vec<GroupNode> {
    if flags.contains('q') {
        return Vec::new();
    }
    let source: Vec<char> = if flags.contains('x') {
        strip_extended_whitespace(pattern)
    } else {
        pattern.chars().collect()
    };
    let mut roots: Vec<GroupNode> = Vec::new();
    let mut stack: Vec<GroupNode> = Vec::new();
    let mut next_number = 1_usize;
    let mut class_depth = 0_usize;
    let mut index = 0_usize;
    while index < source.len() {
        let character = source[index];
        index += 1;
        match character {
            '\\' => index += 1,
            '[' if class_depth > 0 => class_depth += 1,
            '[' => class_depth = 1,
            ']' if class_depth > 0 => class_depth -= 1,
            _ if class_depth > 0 => {}
            '(' => {
                if source.get(index) == Some(&'?') {
                    // `(?:`, and any other extension the dialect may add, captures nothing.
                    // The parenthesis still has to be balanced, so push a sentinel whose
                    // children are re-parented to the enclosing level when it closes.
                    stack.push(GroupNode {
                        number: 0,
                        children: Vec::new(),
                    });
                } else {
                    stack.push(GroupNode {
                        number: next_number,
                        children: Vec::new(),
                    });
                    next_number += 1;
                }
            }
            ')' => {
                if let Some(finished) = stack.pop() {
                    let target = stack
                        .last_mut()
                        .map_or(&mut roots, |open| &mut open.children);
                    if finished.number == 0 {
                        target.extend(finished.children);
                    } else {
                        target.push(finished);
                    }
                }
            }
            _ => {}
        }
    }
    // An unbalanced pattern never reaches here — the regex engine rejects it with FORX0002 —
    // but closing what is left keeps this function total.
    while let Some(finished) = stack.pop() {
        let target = stack
            .last_mut()
            .map_or(&mut roots, |open| &mut open.children);
        if finished.number == 0 {
            target.extend(finished.children);
        } else {
            target.push(finished);
        }
    }
    roots
}

/// Removes the whitespace the `x` flag ignores: whitespace outside a character class, and not
/// escaped by a backslash.
fn strip_extended_whitespace(pattern: &str) -> Vec<char> {
    let source: Vec<char> = pattern.chars().collect();
    let mut out = Vec::with_capacity(source.len());
    let mut class_depth = 0_usize;
    let mut index = 0_usize;
    while index < source.len() {
        let character = source[index];
        index += 1;
        match character {
            '\\' => {
                out.push(character);
                if let Some(next) = source.get(index) {
                    out.push(*next);
                    index += 1;
                }
            }
            '[' => {
                class_depth += 1;
                out.push(character);
            }
            ']' if class_depth > 0 => {
                class_depth -= 1;
                out.push(character);
            }
            c if class_depth == 0 && c.is_whitespace() => {}
            c => out.push(c),
        }
    }
    out
}

/// The result of `fn:analyze-string`: the `fn:analyze-string-result` element node.
///
/// `input` is the string to analyse — the caller has already replaced an empty sequence with
/// the zero-length string, as §5.6.5 requires.
///
/// # Errors
///
/// * [`FnErrorCode::Forx0001`] for invalid flags and [`FnErrorCode::Forx0002`] for an invalid
///   pattern, both raised by the regular-expression engine;
/// * [`FnErrorCode::Forx0003`] if the pattern matches a zero-length string, which §5.6.5 makes
///   an error for this function as it does for `fn:replace` and `fn:tokenize`.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::analyze_string;
/// use oxixml_infoset::XdmNode;
///
/// let result = analyze_string("banana", "a(n)", "")?;
/// assert_eq!(result.string_value(), "banana");
/// assert_eq!(result.children().len(), 4);
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn analyze_string(input: &str, pattern: &str, flags: &str) -> FnResult<NodeRef> {
    let regex = Regex::xpath(pattern, flags)?;
    // §5.6.5: "It is a dynamic error if the regular expression matches a zero-length string."
    // The engine keeps that predicate private, but it is exactly "does the pattern match the
    // empty input", which `find` answers.
    if regex.find("")?.is_some() {
        return Err(FnError::new(
            FnErrorCode::Forx0003,
            "the pattern matches a zero-length string",
        ));
    }
    let forest = group_forest(pattern, flags);
    build_result(input, &regex, &forest).map_err(FnError::from)
}

/// Builds the result tree. Split out so that the tree-building errors — which cannot happen for
/// a well-formed traversal — convert in one place.
fn build_result(input: &str, regex: &Regex, forest: &[GroupNode]) -> Result<NodeRef, XdmError> {
    let mut builder = TreeBuilder::new();
    builder.start_element(ExpandedName::new(
        Some(FN_NAMESPACE),
        "analyze-string-result",
    ))?;
    builder.declare_namespace(Some(FN_PREFIX), FN_NAMESPACE)?;
    let mut position = 0_usize;
    for found in regex.find_iter(input) {
        // A back-reference pattern can exhaust the back-tracking budget; stopping the scan is
        // the only honest thing to do, and leaves the rest of the input as a non-match.
        let Ok(found) = found else { break };
        let range = found.range();
        if range.start > position {
            emit_non_match(&mut builder, &input[position..range.start])?;
        }
        builder.start_element(ExpandedName::new(Some(FN_NAMESPACE), "match"))?;
        let groups: Vec<Option<Range<usize>>> = found.groups().collect();
        emit_content(&mut builder, input, range.clone(), forest, &groups)?;
        builder.end_element()?;
        position = range.end;
    }
    if position < input.len() {
        emit_non_match(&mut builder, &input[position..])?;
    }
    builder.end_element()?;
    let tree = builder.build()?;
    tree.document_element().ok_or_else(|| {
        XdmError::new(
            oxixml_infoset::ErrorCode::Fodc0006,
            "the analyze-string result has no document element",
        )
    })
}

fn emit_non_match(builder: &mut TreeBuilder, text: &str) -> Result<(), XdmError> {
    if text.is_empty() {
        return Ok(());
    }
    builder.start_element(ExpandedName::new(Some(FN_NAMESPACE), "non-match"))?;
    builder.text(text)?;
    builder.end_element()?;
    Ok(())
}

/// Emits the content of one `fn:match` or one `fn:group`: the groups nested directly inside
/// `level` that participated within `range`, in position order, with the uncaptured text
/// between them.
fn emit_content(
    builder: &mut TreeBuilder,
    input: &str,
    range: Range<usize>,
    level: &[GroupNode],
    groups: &[Option<Range<usize>>],
) -> Result<(), XdmError> {
    let mut participating: Vec<(&GroupNode, Range<usize>)> = level
        .iter()
        .filter_map(|node| {
            let captured = groups.get(node.number - 1)?.clone()?;
            (captured.start >= range.start && captured.end <= range.end).then_some((node, captured))
        })
        .collect();
    // Siblings in the pattern are disjoint within one match, so ordering them by start position
    // is the document order of their content. A tie can only arise between two zero-length
    // captures, and their pattern order is then the right one.
    participating.sort_by_key(|(node, captured)| (captured.start, node.number));
    let mut position = range.start;
    for (node, captured) in participating {
        if captured.start < position {
            // A repeated group whose last occurrence lies before text already emitted cannot be
            // placed; §5.6.5 leaves such a capture unreported rather than reordering the text.
            continue;
        }
        if captured.start > position {
            builder.text(&input[position..captured.start])?;
        }
        builder.start_element(ExpandedName::new(Some(FN_NAMESPACE), "group"))?;
        builder.attribute(
            ExpandedName::new(None::<&str>, "nr"),
            node.number.to_string(),
        )?;
        emit_content(builder, input, captured.clone(), &node.children, groups)?;
        builder.end_element()?;
        position = captured.end;
    }
    if position < range.end {
        builder.text(&input[position..range.end])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xmlio::{SerializationParameters, serialize};
    use oxixml_infoset::Sequence;

    fn analyzed(input: &str, pattern: &str, flags: &str) -> String {
        let node = analyze_string(input, pattern, flags).expect("analyze-string succeeds");
        serialize(&Sequence::from(node), &SerializationParameters::default())
            .expect("the result serializes")
    }

    #[test]
    fn group_nesting_is_read_from_the_pattern_not_the_positions() {
        assert_eq!(
            group_forest("(b)(x?)", ""),
            vec![
                GroupNode {
                    number: 1,
                    children: Vec::new()
                },
                GroupNode {
                    number: 2,
                    children: Vec::new()
                },
            ]
        );
        assert_eq!(
            group_forest("(b(x?))", ""),
            vec![GroupNode {
                number: 1,
                children: vec![GroupNode {
                    number: 2,
                    children: Vec::new()
                }],
            }]
        );
    }

    #[test]
    fn non_capturing_groups_and_classes_do_not_count() {
        assert_eq!(
            group_forest("(?:b(an)*a)", ""),
            vec![GroupNode {
                number: 1,
                children: Vec::new()
            }]
        );
        assert_eq!(group_forest("[()]", ""), Vec::new());
        assert_eq!(group_forest("\\(a\\)", ""), Vec::new());
        assert_eq!(group_forest("(a)", "q"), Vec::new());
        assert_eq!(
            group_forest(" ( ?: (a) ) ", "x"),
            vec![GroupNode {
                number: 1,
                children: Vec::new()
            }]
        );
    }

    #[test]
    fn character_class_subtraction_nests() {
        assert_eq!(group_forest("[a-z-[(]]", ""), Vec::new());
    }

    #[test]
    fn the_specification_examples_are_reproduced() {
        assert_eq!(
            analyzed("banana", "a", ""),
            "<fn:analyze-string-result xmlns:fn=\"http://www.w3.org/2005/xpath-functions\">\
             <fn:non-match>b</fn:non-match><fn:match>a</fn:match>\
             <fn:non-match>n</fn:non-match><fn:match>a</fn:match>\
             <fn:non-match>n</fn:non-match><fn:match>a</fn:match>\
             </fn:analyze-string-result>"
        );
        assert_eq!(
            analyzed("banana", "a(n)", ""),
            "<fn:analyze-string-result xmlns:fn=\"http://www.w3.org/2005/xpath-functions\">\
             <fn:non-match>b</fn:non-match><fn:match>a<fn:group nr=\"1\">n</fn:group></fn:match>\
             <fn:match>a<fn:group nr=\"1\">n</fn:group></fn:match>\
             <fn:non-match>a</fn:non-match></fn:analyze-string-result>"
        );
    }

    #[test]
    fn an_empty_input_yields_an_empty_result_element() {
        assert_eq!(
            analyzed("", "abc", ""),
            "<fn:analyze-string-result xmlns:fn=\"http://www.w3.org/2005/xpath-functions\"/>"
        );
    }

    #[test]
    fn the_error_conditions_of_the_specification_hold() {
        assert!(
            analyze_string("abc", ")-(", "")
                .expect_err("invalid pattern")
                .is(FnErrorCode::Forx0002)
        );
        assert!(
            analyze_string("abc", "a", "z")
                .expect_err("invalid flags")
                .is(FnErrorCode::Forx0001)
        );
        assert!(
            analyze_string("abc", "a*", "")
                .expect_err("matches the empty string")
                .is(FnErrorCode::Forx0003)
        );
    }
}
