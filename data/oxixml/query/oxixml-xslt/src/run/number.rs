//! `xsl:number`: computing a sequence of numbers from a node's position, and formatting it.
//!
//! [XSLT 3.0 §12](https://www.w3.org/TR/xslt-30/#number) splits the instruction in two. The
//! *place marker* is the list of integers the `level`, `count` and `from` attributes select —
//! one integer for `single` and `any`, one per ancestor level for `multiple`. The *format* then
//! turns that list into a string, alternating format tokens (`1`, `01`, `a`, `A`, `i`, `I`,
//! `w`, `W`, `Ww`) with the separators written between them.

use super::{Context, Transformer};
use crate::compile::{NumberLevel, NumberSpec};
use crate::error::XsltResult;
use oxixml_infoset::{ExpandedName, NodeKind, NodeRef, XdmNode};

mod count;

pub(crate) use count::Count;

impl Transformer {
    /// Runs one `xsl:number`.
    ///
    /// # Errors
    ///
    /// Any dynamic error the `value`, `count` or `from` expressions raise.
    pub(crate) fn format_number_instruction(
        &mut self,
        spec: &NumberSpec,
        context: &Context,
    ) -> XsltResult<String> {
        let numbers = match &spec.value {
            Some(value) => {
                let sequence = self.evaluate(value, context)?.into_sequence()?.atomize()?;
                let backwards = self.ir().version == crate::compile::XsltVersion::V10;
                let mut out = Vec::with_capacity(sequence.len());
                for item in sequence.iter() {
                    let number = item
                        .as_atomic()
                        .and_then(oxixml_infoset::AtomicValue::as_f64)
                        .or_else(|| item.string_value().ok()?.trim().parse::<f64>().ok())
                        .unwrap_or(f64::NAN);
                    // XSLT 3.0 §12.3: an item that cannot be converted to a non-negative
                    // integer is `XTDE0980`. XSLT 1.0 behaviour instead formats the value as
                    // it stands, which is why the version decides.
                    if number.is_nan() || number.round() < 0.0 {
                        if backwards {
                            return Ok(item.string_value()?);
                        }
                        return Err(crate::error::XsltError::dynamic(
                            "XTDE0980",
                            format!(
                                "`{}` is not a non-negative integer",
                                item.string_value().unwrap_or_default()
                            ),
                        ));
                    }
                    // An `xs:integer` the source wrote out keeps every digit it was written
                    // with; anything else is the exact integer its double value casts to.
                    out.push(
                        item.string_value()
                            .ok()
                            .as_deref()
                            .and_then(Count::parse_integer)
                            .unwrap_or_else(|| Count::from_f64(number)),
                    );
                }
                out
            }
            None => {
                let node = match &spec.select {
                    Some(select) => {
                        // XSLT 3.0 §12.1: the `select` expression must select exactly one node.
                        let value = self.evaluate(select, context)?;
                        let nodes = value.nodes();
                        if nodes.len() != 1 || nodes.len() != value.len() {
                            return Err(crate::error::XsltError::dynamic(
                                "XTTE1000",
                                "the select expression of xsl:number must select a single node",
                            ));
                        }
                        nodes.into_iter().next()
                    }
                    None => {
                        // XSLT 3.0 §12.1: without `value` or `select` the context item must be
                        // a node.
                        let node = context
                            .item
                            .as_ref()
                            .and_then(|item| item.as_node().cloned());
                        if node.is_none() {
                            return Err(crate::error::XsltError::dynamic(
                                "XTTE0990",
                                "xsl:number has no value or select attribute and the context item is not a node",
                            ));
                        }
                        node
                    }
                };
                match node {
                    // The counting forms number positions within a tree, which no machine
                    // integer could run out of room for.
                    Some(node) => self
                        .place_marker(spec, &node, context)?
                        .into_iter()
                        .map(Count::from)
                        .collect(),
                    None => Vec::new(),
                }
            }
        };

        let language = match &spec.language {
            Some(avt) => {
                let language = self.evaluate_avt(avt, context)?;
                if !crate::names::is_language(language.trim()) {
                    return Err(crate::error::XsltError::dynamic(
                        "XTDE0030",
                        format!("`{language}` is not a value for the `lang` attribute"),
                    ));
                }
                Some(language)
            }
            None => None,
        };
        let format = self.evaluate_avt(&spec.format, context)?;
        let letter_value = match &spec.letter_value {
            Some(avt) => Some(self.evaluate_avt(avt, context)?),
            None => None,
        };
        let grouping_separator = match &spec.grouping_separator {
            Some(avt) => self.evaluate_avt(avt, context)?.chars().next(),
            None => None,
        };
        let grouping_size = match &spec.grouping_size {
            Some(avt) => self
                .evaluate_avt(avt, context)?
                .trim()
                .parse::<usize>()
                .ok(),
            None => None,
        };
        let ordinal = match &spec.ordinal {
            Some(avt) => Some(self.evaluate_avt(avt, context)?),
            None => None,
        };
        // `start-at` shifts the whole place marker, so that `start-at="0"` numbers from zero.
        let numbers = match &spec.start_at {
            Some(avt) => {
                let starts: Vec<i64> = self
                    .evaluate_avt(avt, context)?
                    .split_whitespace()
                    .map(|token| token.parse::<i64>().unwrap_or(1))
                    .collect();
                let mut shifted = Vec::with_capacity(numbers.len());
                for (level, value) in numbers.into_iter().enumerate() {
                    let start = starts
                        .get(level)
                        .copied()
                        .or_else(|| starts.last().copied())
                        .unwrap_or(1);
                    // The shift is `value + start - 1`, computed so that a `start-at` near the
                    // ends of the integer range reports the condition rather than wrapping: an
                    // unrepresentable number is not one this instruction can format.
                    let Some(number) = value.shifted_to_start(start) else {
                        return Err(crate::error::XsltError::dynamic(
                            "XTDE0980",
                            format!(
                                "shifting the place marker {value} by `start-at=\"{start}\"` \
                                 leaves the range of integers this processor can format"
                            ),
                        ));
                    };
                    shifted.push(number);
                }
                shifted
            }
            None => numbers,
        };
        Ok(format_numbers(
            &numbers,
            &format,
            letter_value.as_deref(),
            grouping_separator,
            grouping_size,
            ordinal.as_deref(),
            language.as_deref(),
        ))
    }

    /// The list of integers `level`, `count` and `from` select.
    fn place_marker(
        &mut self,
        spec: &NumberSpec,
        node: &NodeRef,
        context: &Context,
    ) -> XsltResult<Vec<i64>> {
        // XSLT 3.0 §18.4: a pattern is evaluated with the captured substrings of
        // `fn:regex-group` empty, so `count`/`from` inside an `xsl:analyze-string` see none.
        self.without_regex_groups(|engine| engine.place_marker_within(spec, node, context))
    }

    /// The place marker, with the captured-substring set already emptied.
    fn place_marker_within(
        &mut self,
        spec: &NumberSpec,
        node: &NodeRef,
        context: &Context,
    ) -> XsltResult<Vec<i64>> {
        let dynamic = self.pattern_context_for(context);
        let default = default_count_pattern(node);
        let matches_count = |candidate: &NodeRef| -> XsltResult<bool> {
            match &spec.count {
                Some(pattern) => pattern.matches(candidate, &dynamic),
                None => Ok(default.as_ref().is_some_and(|wanted| {
                    candidate.node_kind() == wanted.0 && candidate.node_name() == wanted.1
                })),
            }
        };
        // [XSLT 3.0 §12.3](https://www.w3.org/TR/xslt-30/#number-level-single) defines
        // `matches-from($node)` as "true if and only if the given node matches the pattern given
        // in the `from` attribute, **or if `$node` is the root node of a tree**" — so the
        // boundary always exists, whether or not a `from` attribute was written.
        let matches_from = |candidate: &NodeRef| -> XsltResult<bool> {
            if candidate.parent().is_none() {
                return Ok(true);
            }
            match &spec.from {
                Some(pattern) => pattern.matches(candidate, &dynamic),
                None => Ok(false),
            }
        };

        match spec.level {
            // §12.3 selects `$A`, the innermost ancestor-or-self matching `count`, and `$F`, the
            // innermost matching `from`, and keeps `$A` when `$F` is an **ancestor-or-self** of
            // it. A node that matches both patterns is therefore numbered rather than cut off:
            // `number-3227` numbers the `a` elements it counts under `from="a"`.
            NumberLevel::Single => {
                let counted = ancestors_matching(node, &matches_count)?;
                let boundary = ancestors_matching(node, &matches_from)?;
                let (Some(target), Some(boundary)) = (counted.first(), boundary.first()) else {
                    return Ok(Vec::new());
                };
                if !within(target, boundary) {
                    return Ok(Vec::new());
                }
                Ok(vec![sibling_position(target, &matches_count)?])
            }
            // The same boundary, applied to *every* ancestor-or-self that matches `count`
            // rather than only the innermost: `number-4001` counts a `c` that is also the `from`
            // node, and the levels below it, but not the `b` above it.
            NumberLevel::Multiple => {
                let Some(boundary) = ancestors_matching(node, &matches_from)?.into_iter().next()
                else {
                    return Ok(Vec::new());
                };
                let mut levels = Vec::new();
                for candidate in ancestors_matching(node, &matches_count)? {
                    if within(&candidate, &boundary) {
                        levels.push(sibling_position(&candidate, &matches_count)?);
                    }
                }
                levels.reverse();
                Ok(levels)
            }
            NumberLevel::Any => {
                // [XSLT 3.0 §12.2](https://www.w3.org/TR/xslt-30/#number-level-any): the
                // candidate set is the union of the `preceding` and `ancestor-or-self` axes —
                // so attribute and namespace nodes other than the node being numbered are not
                // considered. `from` truncates the set at the *last* preceding node that it
                // matches; when nothing matches, the whole set counts.
                let root = node.root();
                let mut count = 0_i64;
                let mut found = false;
                walk(&root, &mut |candidate: &NodeRef| -> XsltResult<bool> {
                    if found {
                        return Ok(false);
                    }
                    let is_self = candidate == node;
                    let usable = is_self || !matches!(candidate.node_kind(), NodeKind::Attribute);
                    if usable {
                        if matches_from(candidate)? {
                            count = 0;
                        }
                        if matches_count(candidate)? {
                            count += 1;
                        }
                    }
                    if is_self {
                        found = true;
                        return Ok(false);
                    }
                    Ok(true)
                })?;
                Ok(if count == 0 { Vec::new() } else { vec![count] })
            }
        }
    }
}

/// The node kind and name a missing `count` attribute implies.
fn default_count_pattern(node: &NodeRef) -> Option<(NodeKind, Option<ExpandedName>)> {
    Some((node.node_kind(), node.node_name()))
}

/// Walks a tree in document order, stopping when the visitor returns `false`.
fn walk(node: &NodeRef, visit: &mut dyn FnMut(&NodeRef) -> XsltResult<bool>) -> XsltResult<bool> {
    if !visit(node)? {
        return Ok(false);
    }
    for attribute in node.attributes() {
        if !visit(&attribute)? {
            return Ok(false);
        }
    }
    for child in node.children() {
        if !walk(&child, visit)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The ancestor-or-self nodes of `node` that `matches` accepts, innermost first.
fn ancestors_matching(
    node: &NodeRef,
    matches: &dyn Fn(&NodeRef) -> XsltResult<bool>,
) -> XsltResult<Vec<NodeRef>> {
    let mut found = Vec::new();
    let mut current = Some(node.clone());
    while let Some(candidate) = current {
        if matches(&candidate)? {
            found.push(candidate.clone());
        }
        current = candidate.parent();
    }
    Ok(found)
}

/// Whether `node` is inside the subtree rooted at `boundary` — the `ancestor-or-self` test
/// `$A[ancestor-or-self::node()[. is $F]]` of [XSLT 3.0
/// §12.3](https://www.w3.org/TR/xslt-30/#number-level-single), which admits the boundary node
/// itself.
fn within(node: &NodeRef, boundary: &NodeRef) -> bool {
    let mut current = Some(node.clone());
    while let Some(candidate) = current {
        if &candidate == boundary {
            return true;
        }
        current = candidate.parent();
    }
    false
}

/// One more than the number of preceding siblings the `count` pattern matches.
fn sibling_position(
    node: &NodeRef,
    matches_count: &dyn Fn(&NodeRef) -> XsltResult<bool>,
) -> XsltResult<i64> {
    let Some(parent) = node.parent() else {
        return Ok(1);
    };
    let mut position = 0_i64;
    for sibling in parent.children() {
        if matches_count(&sibling)? {
            position += 1;
        }
        if &sibling == node {
            return Ok(position.max(1));
        }
    }
    Ok(position.max(1))
}

/// One token of a `format` attribute.
#[derive(Debug, PartialEq, Eq)]
enum Token {
    /// A run of separator characters.
    Separator(String),
    /// A numbering token, such as `1`, `01`, `a`, `I` or `Ww`.
    Format(String),
}

/// Splits a `format` attribute into alternating separator and format tokens.
fn tokenize(format: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut current_is_format = None;
    for character in format.chars() {
        let is_format = character.is_alphanumeric();
        if current_is_format != Some(is_format) && !current.is_empty() {
            tokens.push(match current_is_format {
                Some(true) => Token::Format(std::mem::take(&mut current)),
                _ => Token::Separator(std::mem::take(&mut current)),
            });
        }
        current_is_format = Some(is_format);
        current.push(character);
    }
    if !current.is_empty() {
        tokens.push(match current_is_format {
            Some(true) => Token::Format(current),
            _ => Token::Separator(current),
        });
    }
    tokens
}

/// Formats a place marker under a `format` attribute.
#[must_use]
pub(crate) fn format_numbers(
    numbers: &[Count],
    format: &str,
    letter_value: Option<&str>,
    grouping_separator: Option<char>,
    grouping_size: Option<usize>,
    ordinal: Option<&str>,
    language: Option<&str>,
) -> String {
    let tokens = tokenize(if format.is_empty() { "1" } else { format });
    // [XSLT 3.0 §12.3](https://www.w3.org/TR/xslt-30/#number-formats): the characters before
    // the first format token are the prefix and those after the last are the suffix. When the
    // format contains no format token at all the whole string is *both*, and the default
    // token `1` supplies the numbering.
    let default_token = "1".to_owned();
    let formats: Vec<&String> = tokens
        .iter()
        .filter_map(|token| match token {
            Token::Format(text) => Some(text),
            Token::Separator(_) => None,
        })
        .collect();
    let (prefix, suffix, formats) = if formats.is_empty() {
        let whole = match tokens.first() {
            Some(Token::Separator(text)) => text.clone(),
            _ => String::new(),
        };
        (whole.clone(), whole, vec![&default_token])
    } else {
        let prefix = match tokens.first() {
            Some(Token::Separator(text)) => text.clone(),
            _ => String::new(),
        };
        let suffix = match tokens.last() {
            Some(Token::Separator(text)) if tokens.len() > 1 => text.clone(),
            _ => String::new(),
        };
        (prefix, suffix, formats)
    };
    // An empty place marker still contributes the prefix and the suffix: `format="A."` on a
    // node that matches nothing produces `.`, as the suite's `number-32xx` cases require.
    if numbers.is_empty() {
        return format!("{prefix}{suffix}");
    }
    let separators: Vec<String> = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| match token {
            Token::Separator(text)
                if index != 0 && !(index + 1 == tokens.len() && tokens.len() > 1) =>
            {
                Some(text.clone())
            }
            _ => None,
        })
        .collect();

    let mut out = prefix;
    for (index, number) in numbers.iter().enumerate() {
        if index > 0 {
            let separator = separators
                .get(index - 1)
                .cloned()
                .or_else(|| separators.last().cloned())
                .unwrap_or_else(|| ".".to_owned());
            out.push_str(&separator);
        }
        let token = formats
            .get(index)
            .copied()
            .or_else(|| formats.last().copied())
            .map_or("1", String::as_str);
        out.push_str(&format_one(
            number,
            token,
            letter_value,
            grouping_separator,
            grouping_size,
            ordinal,
            language,
        ));
    }
    out.push_str(&suffix);
    out
}

fn format_one(
    number: &Count,
    token: &str,
    letter_value: Option<&str>,
    grouping_separator: Option<char>,
    grouping_size: Option<usize>,
    ordinal: Option<&str>,
    language: Option<&str>,
) -> String {
    let alphabetic = letter_value == Some("alphabetic");
    let first = token.chars().next().unwrap_or('1');
    let language = NumberLanguage::parse(language);
    let width = token.chars().count();
    // A sequence spelled out of a fixed alphabet — Roman numerals, words, the circled digits —
    // is defined only over the values a machine integer holds, and
    // [XSLT 3.0 §12.3](https://www.w3.org/TR/xslt-30/#number-formats) lets a processor fall back
    // on decimal notation "if the numbering sequence does not extend far enough". The decimal
    // forms below have no such limit and take the value whole.
    let sequenced = |spell: &dyn Fn(i64) -> String| match number.as_i64() {
        Some(small) => spell(small),
        None => decimal(number, width, grouping_separator, grouping_size),
    };
    match first {
        'i' if !alphabetic => sequenced(&|value| roman(value, false)),
        'I' if !alphabetic => sequenced(&|value| roman(value, true)),
        'w' => sequenced(&|value| words(value, WordCase::Lower, ordinal, language)),
        'W' if token == "Ww" => {
            sequenced(&|value| words(value, WordCase::Title, ordinal, language))
        }
        'W' => sequenced(&|value| words(value, WordCase::Upper, ordinal, language)),
        'a' => sequenced(&|value| alphabetic_number(value, false)),
        'A' => sequenced(&|value| alphabetic_number(value, true)),
        _ if numbering_sequence(first).is_some() => {
            let Some(sequence) = numbering_sequence(first) else {
                return number.to_string();
            };
            sequenced(&|value| sequence.format(value))
        }
        _ if alphabet_for(first).is_some() => {
            let Some(alphabet) = alphabet_for(first) else {
                return number.to_string();
            };
            sequenced(&|value| alphabetic_in(value, &alphabet))
        }
        _ if digit_family(first).is_some_and(|zero| zero != u32::from(b'0')) => {
            // A digit from another Unicode decimal family: the number is written exactly as the
            // ASCII form would be — same width, same grouping, same sign — and each digit is then
            // mapped into that family, which leaves the separators and the sign alone.
            let Some(zero) = digit_family(first).and_then(char::from_u32) else {
                return number.to_string();
            };
            decimal(number, width, grouping_separator, grouping_size)
                .chars()
                .map(|digit| {
                    digit
                        .to_digit(10)
                        .and_then(|value| char::from_u32(zero as u32 + value))
                        .unwrap_or(digit)
                })
                .collect()
        }
        _ if first.is_ascii_digit() => {
            let digits = decimal(number, width, grouping_separator, grouping_size);
            match ordinal {
                Some(spec) if !spec.is_empty() && spec != "no" => {
                    format!("{digits}{}", ordinal_suffix(number))
                }
                _ => digits,
            }
        }
        _ => number.to_string(),
    }
}

fn decimal(
    number: &Count,
    width: usize,
    grouping_separator: Option<char>,
    grouping_size: Option<usize>,
) -> String {
    // The magnitude is taken unsigned so that the smallest representable integer, whose negation
    // does not fit, still prints one sign rather than two.
    let digits = format!("{:0>width$}", number.magnitude(), width = width.max(1));
    let grouped = match (grouping_separator, grouping_size) {
        (Some(separator), Some(size)) if size > 0 => {
            let characters: Vec<char> = digits.chars().collect();
            let mut out = String::with_capacity(characters.len() * 2);
            for (index, character) in characters.iter().enumerate() {
                if index > 0 && (characters.len() - index) % size == 0 {
                    out.push(separator);
                }
                out.push(*character);
            }
            out
        }
        _ => digits,
    };
    if number.is_negative() {
        return format!("-{grouped}");
    }
    grouped
}

fn ordinal_suffix(number: &Count) -> &'static str {
    let magnitude = number.magnitude();
    let digits = magnitude.as_bytes();
    let last = digits.last().map_or(0, |byte| byte - b'0');
    let tens = digits
        .len()
        .checked_sub(2)
        .and_then(|index| digits.get(index))
        .map_or(0, |byte| byte - b'0');
    match (i64::from(tens) * 10 + i64::from(last), i64::from(last)) {
        (11..=13, _) => "th",
        (_, 1) => "st",
        (_, 2) => "nd",
        (_, 3) => "rd",
        _ => "th",
    }
}

/// A family of code points that spells out consecutive integers — the *numbering sequences* of
/// [XSLT 3.0 §12.3](https://www.w3.org/TR/xslt-30/#number-formats).
///
/// A family is not always one contiguous block: the circled digits run `①`–`⑳` at `U+2460`,
/// continue with `㉑`–`㉟` at `U+3251` and finish with `㊱`–`㊿` at `U+32B1`, so a sequence is
/// described by a list of runs. Values outside every run fall back to decimal, which is what
/// the "overflowing range" cases of the suite expect.
struct NumberingSequence {
    /// `(first code point, first integer, last integer)` for each contiguous run.
    runs: &'static [(u32, i64, i64)],
    /// The code point for zero, when the family has one.
    zero: Option<u32>,
}

impl NumberingSequence {
    fn format(&self, number: i64) -> String {
        if number == 0
            && let Some(zero) = self.zero
            && let Some(character) = char::from_u32(zero)
        {
            return character.to_string();
        }
        for (start, first_value, last_value) in self.runs.iter().copied() {
            if number < first_value || number > last_value {
                continue;
            }
            let offset = (number - first_value) as u32;
            return char::from_u32(start + offset).map_or_else(|| number.to_string(), String::from);
        }
        number.to_string()
    }
}

/// The zero of the Unicode decimal digit family `first` belongs to, if any.
///
/// [`char::to_digit`] only understands ASCII, so the families a `format` attribute may name are
/// listed here; each is a run of ten consecutive code points starting at its zero.
fn digit_family(first: char) -> Option<u32> {
    const ZEROS: &[u32] = &[
        0x0030, // ASCII
        0x0660, // Arabic-Indic
        0x06F0, // Extended Arabic-Indic
        0x0966, // Devanagari
        0x09E6, // Bengali
        0x0A66, // Gurmukhi
        0x0AE6, // Gujarati
        0x0B66, // Oriya
        0x0BE6, // Tamil
        0x0C66, // Telugu
        0x0CE6, // Kannada
        0x0D66, // Malayalam
        0x0E50, // Thai
        0x0ED0, // Lao
        0x0F20, // Tibetan
        0x1040, // Myanmar
        0x17E0, // Khmer
        0x1810, // Mongolian
        0xFF10, // Fullwidth
    ];
    let code = first as u32;
    ZEROS
        .iter()
        .copied()
        .find(|zero| code >= *zero && code < zero + 10)
}

/// The `(first code point, first integer, last integer)` runs a numbering sequence is made of.
type SequenceRuns = &'static [(u32, i64, i64)];

/// Every numbering sequence this processor recognizes, as `(runs, zero)`.
const NUMBERING_SEQUENCES: &[(SequenceRuns, Option<u32>)] = &[
    // ①..⑳, ㉑..㉟, ㊱..㊿, with ⓪ for zero.
    (
        &[(0x2460, 1, 20), (0x3251, 21, 35), (0x32B1, 36, 50)],
        Some(0x24EA),
    ),
    // ⑴..⒇ — parenthesized digits, which have no zero.
    (&[(0x2474, 1, 20)], None),
    // ⒈..⒛ — digits followed by a full stop, with 🄀 for zero.
    (&[(0x2488, 1, 20)], Some(0x1F100)),
    // ⑴..⒵ — parenthesized Latin small letters.
    (&[(0x249C, 1, 26)], None),
    // Ⓐ..Ⓩ.
    (&[(0x24B6, 1, 26)], None),
    // ⓐ..ⓩ.
    (&[(0x24D0, 1, 26)], None),
    // ❶..❿ continuing with ⓫..⓴, with ⓿ for zero.
    (&[(0x2776, 1, 10), (0x24EB, 11, 20)], Some(0x24FF)),
    // ⓵..⓾ — double circled digits.
    (&[(0x24F5, 1, 10)], None),
    // ➀..➉ — dingbat circled sans-serif digits, with 🄋 for zero.
    (&[(0x2780, 1, 10)], Some(0x1F10B)),
    // ➊..➓ — dingbat negative circled sans-serif digits, with 🄌 for zero.
    (&[(0x278A, 1, 10)], Some(0x1F10C)),
    // ㈠..㈩ — parenthesized ideographs.
    (&[(0x3220, 1, 10)], None),
    // ㊀..㊉ — circled ideographs.
    (&[(0x3280, 1, 10)], None),
    // 𐄇..𐄐 — Aegean numbers.
    (&[(0x10107, 1, 10)], None),
    // 𐋡..𐋪 — Coptic epact digits.
    (&[(0x102E1, 1, 10)], None),
    // 𐹠..𐹩 — Rumi digits.
    (&[(0x10E60, 1, 10)], None),
    // 𑁒..𑁛 — Brahmi numbers.
    (&[(0x11052, 1, 10)], None),
    // 𑇡..𑇪 — Sinhala archaic numbers.
    (&[(0x111E1, 1, 10)], None),
    // 𝍠..𝍨 — counting rod unit digits.
    (&[(0x1D360, 1, 9)], None),
    // 𞣇..𞣏 — Mende Kikakui digits.
    (&[(0x1E8C7, 1, 9)], None),
    // 🄂..🄊 — digits followed by a comma, with 🄁 for zero.
    (&[(0x1F102, 1, 9)], Some(0x1F101)),
];

/// The numbering sequence a format token belongs to, if it belongs to one.
fn numbering_sequence(first: char) -> Option<NumberingSequence> {
    let code = first as u32;
    for (runs, zero) in NUMBERING_SEQUENCES.iter().copied() {
        let inside = runs.iter().any(|(start, first_value, last_value)| {
            let length = (last_value - first_value + 1) as u32;
            code >= *start && code < start + length
        });
        if inside || zero == Some(code) {
            return Some(NumberingSequence { runs, zero });
        }
    }
    None
}

const ROMAN: &[(i64, &str)] = &[
    (1000, "m"),
    (900, "cm"),
    (500, "d"),
    (400, "cd"),
    (100, "c"),
    (90, "xc"),
    (50, "l"),
    (40, "xl"),
    (10, "x"),
    (9, "ix"),
    (5, "v"),
    (4, "iv"),
    (1, "i"),
];

/// The first number Roman notation no longer spells.
///
/// [XSLT 3.0 §12.3](https://www.w3.org/TR/xslt-30/#number) lets a processor fall back on decimal
/// notation "if the numbering sequence does not extend far enough", and a hundred thousand `m`s
/// is not a numbering sequence anyone reads: `number-0603` writes `100000` with `format="i"` and
/// expects the digits back.
const ROMAN_CEILING: i64 = 100_000;

fn roman(number: i64, upper: bool) -> String {
    if number <= 0 || number >= ROMAN_CEILING {
        return number.to_string();
    }
    let mut remaining = number;
    let mut out = String::new();
    for (value, numeral) in ROMAN {
        while remaining >= *value {
            out.push_str(numeral);
            remaining -= value;
        }
    }
    if upper { out.to_uppercase() } else { out }
}

fn alphabetic_number(number: i64, upper: bool) -> String {
    const LOWER: &str = "abcdefghijklmnopqrstuvwxyz";
    const UPPER: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let alphabet: Vec<char> = if upper { UPPER } else { LOWER }.chars().collect();
    alphabetic_in(number, &alphabet)
}

/// The alphabets a format token can start an *alphabetic* numbering sequence in, beyond the
/// Latin `a`–`z` that `format="a"` and `format="A"` name outright.
///
/// [XSLT 3.0 §12.3](https://www.w3.org/TR/xslt-30/#number): "any other format token indicates a
/// numbering sequence that starts with that token", so `format="&#x3b1;"` counts α, β, γ … ω,
/// αα, αβ. `number-0902` asks for exactly that, and `number-0901` asks for it under
/// `letter-value="traditional"` as well: the Greek *numeral* notation that attribute would
/// otherwise select is discretionary — the case's own comment records that no widely used
/// processor implements it — and its expected result is the alphabetic sequence either way.
const ALPHABETS: &[&str] = &[
    // Greek, lower case. Both sigmas are letters of the sequence, which is what the expected
    // result of `number-0902` records: … ρ, ς, σ, τ …
    "αβγδεζηθικλμνξοπρςστυφχψω",
    // Greek, upper case, which has no final sigma.
    "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
];

/// The alphabetic numbering sequence a format token beginning with `first` denotes, rotated so
/// that `first` is the letter numbering starts at.
fn alphabet_for(first: char) -> Option<Vec<char>> {
    let alphabet = ALPHABETS
        .iter()
        .find(|alphabet| alphabet.chars().any(|letter| letter == first))?;
    let mut letters: Vec<char> = alphabet.chars().collect();
    let start = letters.iter().position(|letter| *letter == first)?;
    letters.rotate_left(start);
    Some(letters)
}

/// Numbers in `alphabet`, which counts past its own end the way `a`–`z` counts on into `aa`.
fn alphabetic_in(number: i64, alphabet: &[char]) -> String {
    let base = alphabet.len() as i64;
    if number <= 0 || base == 0 {
        return number.to_string();
    }
    let mut remaining = number;
    let mut out = Vec::new();
    while remaining > 0 {
        remaining -= 1;
        let Some(letter) = alphabet.get((remaining % base) as usize) else {
            return number.to_string();
        };
        out.push(*letter);
        remaining /= base;
    }
    out.reverse();
    out.into_iter().collect()
}

#[derive(Clone, Copy)]
enum WordCase {
    Lower,
    Upper,
    Title,
}

const UNITS: &[&str] = &[
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const TENS: &[&str] = &[
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

const ORDINAL_UNITS: &[&str] = &[
    "zeroth",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
];

const ORDINAL_TENS: &[&str] = &[
    "",
    "",
    "twentieth",
    "thirtieth",
    "fortieth",
    "fiftieth",
    "sixtieth",
    "seventieth",
    "eightieth",
    "ninetieth",
];

fn words(number: i64, case: WordCase, ordinal: Option<&str>, language: NumberLanguage) -> String {
    let spelled = match language {
        NumberLanguage::English => spell(number, ordinal.is_some()),
        NumberLanguage::German => german(number, ordinal),
        NumberLanguage::Italian => italian(number, ordinal),
    };
    match case {
        WordCase::Lower => spelled,
        WordCase::Upper => spelled.to_uppercase(),
        WordCase::Title => title_case(&spelled),
    }
}

fn title_case(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut at_start = true;
    for character in value.chars() {
        if at_start && character.is_alphabetic() {
            out.extend(character.to_uppercase());
            at_start = false;
        } else {
            out.push(character);
            if !character.is_alphabetic() {
                at_start = true;
            }
        }
    }
    out
}

fn spell(number: i64, ordinal: bool) -> String {
    // The magnitude is taken unsigned: the smallest representable integer has no positive
    // counterpart, so negating it here would be the one input this spelling could not carry.
    let magnitude = spell_magnitude(number.unsigned_abs(), ordinal);
    if number < 0 {
        return format!("minus {magnitude}");
    }
    magnitude
}

fn spell_magnitude(number: u64, ordinal: bool) -> String {
    let cardinal = |value: u64| spell_magnitude(value, false);
    match number {
        0..=19 => {
            let table = if ordinal { ORDINAL_UNITS } else { UNITS };
            table.get(number as usize).copied().unwrap_or("").to_owned()
        }
        20..=99 => {
            let tens = (number / 10) as usize;
            let units = number % 10;
            if units == 0 {
                let table = if ordinal { ORDINAL_TENS } else { TENS };
                return table.get(tens).copied().unwrap_or("").to_owned();
            }
            format!(
                "{}-{}",
                TENS.get(tens).copied().unwrap_or(""),
                spell_magnitude(units, ordinal)
            )
        }
        100..=999 => {
            let hundreds = number / 100;
            let rest = number % 100;
            if rest == 0 {
                return format!(
                    "{} hundred{}",
                    cardinal(hundreds),
                    if ordinal { "th" } else { "" }
                );
            }
            format!(
                "{} hundred and {}",
                cardinal(hundreds),
                spell_magnitude(rest, ordinal)
            )
        }
        1_000..=999_999 => scaled(number, 1_000, "thousand", ordinal),
        1_000_000..=999_999_999 => scaled(number, 1_000_000, "million", ordinal),
        _ => scaled(number, 1_000_000_000, "billion", ordinal),
    }
}

/// The language `format="w"` spells a number in.
///
/// [XSLT 3.0 §12.3](https://www.w3.org/TR/xslt-30/#number) leaves the set open: "if the
/// processor does not support numbering in the requested language, it must use a language that
/// it does support", so an unrecognized tag falls back to English rather than failing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NumberLanguage {
    English,
    German,
    Italian,
}

impl NumberLanguage {
    /// The language `tag` names, read from its primary subtag so that `de-AT` spells German.
    fn parse(tag: Option<&str>) -> Self {
        let tag = tag.unwrap_or_default();
        let primary = tag.trim().split(['-', '_']).next().unwrap_or_default();
        match primary.to_ascii_lowercase().as_str() {
            "de" => Self::German,
            "it" => Self::Italian,
            _ => Self::English,
        }
    }
}

const GERMAN_UNITS: &[&str] = &[
    "null",
    "eins",
    "zwei",
    "drei",
    "vier",
    "f\u{fc}nf",
    "sechs",
    "sieben",
    "acht",
    "neun",
];

const GERMAN_TEENS: &[&str] = &[
    "zehn",
    "elf",
    "zw\u{f6}lf",
    "dreizehn",
    "vierzehn",
    "f\u{fc}nfzehn",
    "sechzehn",
    "siebzehn",
    "achtzehn",
    "neunzehn",
];

const GERMAN_TENS: &[&str] = &[
    "",
    "",
    "zwanzig",
    "drei\u{df}ig",
    "vierzig",
    "f\u{fc}nfzig",
    "sechzig",
    "siebzig",
    "achtzig",
    "neunzig",
];

/// The German ordinal stems below twenty, which the ending is attached to.
///
/// Most are the cardinal with `t`; `erst`, `dritt`, `siebt` and `acht` are not, the last because
/// `acht` already ends in the consonant the suffix would add.
const GERMAN_ORDINAL_STEMS: &[&str] = &[
    "nullt",
    "erst",
    "zweit",
    "dritt",
    "viert",
    "f\u{fc}nft",
    "sechst",
    "siebt",
    "acht",
    "neunt",
    "zehnt",
    "elft",
    "zw\u{f6}lft",
    "dreizehnt",
    "vierzehnt",
    "f\u{fc}nfzehnt",
    "sechzehnt",
    "siebzehnt",
    "achtzehnt",
    "neunzehnt",
];

/// Spells `number` in German, as a cardinal or — when `ordinal` is written — as an ordinal.
///
/// `number-0812` and `number-0813` pin the two down between them: `134` is
/// `einhundertvierunddreißig` and, with `ordinal="-es"`, `einhundertvierunddreißigstes`.
fn german(number: i64, ordinal: Option<&str>) -> String {
    let magnitude = number.unsigned_abs();
    let spelled = match ordinal {
        Some(spec) => format!(
            "{}{}",
            german_ordinal_stem(magnitude),
            german_ordinal_ending(spec)
        ),
        None => german_cardinal(magnitude, true),
    };
    if number < 0 {
        return format!("minus {spelled}");
    }
    spelled
}

/// The German cardinal for `number`.
///
/// `standalone` distinguishes the two spellings of one: a number that ends in it is `eins`, and
/// one that is a factor of something larger — `einhundert`, `einundzwanzig` — is `ein`
/// (`number-0812` accepts `zweihundertein` and `zweihunderteins` alike, and `number-2506` wants
/// `Ein` or `Eins`).
fn german_cardinal(number: u64, standalone: bool) -> String {
    let word = |table: &[&str], index: u64| {
        table
            .get(index as usize)
            .copied()
            .unwrap_or_default()
            .to_owned()
    };
    match number {
        1 if !standalone => "ein".to_owned(),
        0..=9 => word(GERMAN_UNITS, number),
        10..=19 => word(GERMAN_TEENS, number - 10),
        20..=99 => {
            let tens = word(GERMAN_TENS, number / 10);
            match number % 10 {
                0 => tens,
                // German writes the unit first, joined by `und`: `vierunddreißig`.
                units => format!("{}und{tens}", german_cardinal(units, false)),
            }
        }
        100..=999 => german_scaled(number, 100, "hundert", ""),
        1_000..=999_999 => german_scaled(number, 1_000, "tausend", ""),
        _ => {
            let millions = number / 1_000_000;
            let name = if millions == 1 {
                " Million"
            } else {
                " Millionen"
            };
            german_scaled(number, 1_000_000, name, " ")
        }
    }
}

/// One scale of a German cardinal: the multiplier, the scale word, and the remainder.
///
/// `separator` is empty for the scales German writes as one word (`zweihundertsechzehn`) and a
/// space for `Million`, which is a noun and stands apart — `number-0812` requires the space in
/// `Zwei Millionen Einhundert…`.
fn german_scaled(number: u64, unit: u64, name: &str, separator: &str) -> String {
    let head = format!("{}{name}", german_cardinal(number / unit, false));
    match number % unit {
        0 => head,
        rest => format!("{head}{separator}{}", german_cardinal(rest, true)),
    }
}

/// The German ordinal stem for `number`, which is the cardinal with only its **last** component
/// made ordinal: `zweihundertzehnt`, not `zweihundertzehnst`.
fn german_ordinal_stem(number: u64) -> String {
    if number < 20 {
        return GERMAN_ORDINAL_STEMS
            .get(number as usize)
            .copied()
            .unwrap_or_default()
            .to_owned();
    }
    let tail = number % 100;
    if tail == 0 {
        return format!("{}st", german_cardinal(number, true));
    }
    let leading = number - tail;
    let head = if leading == 0 {
        String::new()
    } else {
        german_cardinal(leading, false)
    };
    if tail < 20 {
        return format!("{head}{}", german_ordinal_stem(tail));
    }
    format!("{head}{}st", german_cardinal(tail, true))
}

/// The ending an `ordinal` attribute asks a German ordinal stem to take.
///
/// German ordinals decline, so the attribute names the ending rather than a numbering scheme:
/// `number-0813` writes `-e`, `-er`, `-es` and `-en`. A CLDR scheme name such as
/// `%spellout-ordinal` names no ending, and `number-0828` — which appends the declension as
/// literal text — shows the bare `e` is what it must produce.
fn german_ordinal_ending(spec: &str) -> &str {
    match spec.trim() {
        rest if rest.starts_with('-') => rest.trim_start_matches('-'),
        _ => "e",
    }
}

const ITALIAN_UNITS: &[&str] = &[
    "zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove",
];

const ITALIAN_TEENS: &[&str] = &[
    "dieci",
    "undici",
    "dodici",
    "tredici",
    "quattordici",
    "quindici",
    "sedici",
    "diciassette",
    "diciotto",
    "diciannove",
];

const ITALIAN_TENS: &[&str] = &[
    "",
    "",
    "venti",
    "trenta",
    "quaranta",
    "cinquanta",
    "sessanta",
    "settanta",
    "ottanta",
    "novanta",
];

/// The Italian ordinals below eleven, which are irregular; from eleven on the ordinal is the
/// cardinal with its final vowel replaced by `esimo` (`number-0829`).
const ITALIAN_ORDINALS: &[&str] = &[
    "zero", "primo", "secondo", "terzo", "quarto", "quinto", "sesto", "settimo", "ottavo", "nono",
    "decimo",
];

/// Spells `number` in Italian.
///
/// The `ordinal` attribute names a gender as well as a scheme: `number-0829` asks for
/// `%spellout-ordinal-masculine` and `%spellout-ordinal-feminine` over the same ten numbers and
/// expects `primo…decimo` against `prima…decima`.
fn italian(number: i64, ordinal: Option<&str>) -> String {
    let magnitude = number.unsigned_abs();
    let spelled = match ordinal {
        Some(spec) => italian_ordinal(magnitude, spec.contains("feminine")),
        None => italian_cardinal(magnitude),
    };
    if number < 0 {
        return format!("meno {spelled}");
    }
    spelled
}

fn italian_cardinal(number: u64) -> String {
    let word = |table: &[&str], index: u64| {
        table
            .get(index as usize)
            .copied()
            .unwrap_or_default()
            .to_owned()
    };
    match number {
        0..=9 => word(ITALIAN_UNITS, number),
        10..=19 => word(ITALIAN_TEENS, number - 10),
        20..=99 => {
            let tens = word(ITALIAN_TENS, number / 10);
            match number % 10 {
                0 => tens,
                // The tens word drops its final vowel before `uno` and `otto` (`ventuno`,
                // `ventotto`), and a final `tre` takes an accent (`ventitré`).
                units => {
                    let head = if matches!(units, 1 | 8) {
                        tens.trim_end_matches(['a', 'e', 'i', 'o']).to_owned()
                    } else {
                        tens
                    };
                    let tail = if units == 3 {
                        "tr\u{e9}".to_owned()
                    } else {
                        word(ITALIAN_UNITS, units)
                    };
                    format!("{head}{tail}")
                }
            }
        }
        100..=999 => italian_scaled(number, 100, "cento", "cento"),
        1_000..=999_999 => italian_scaled(number, 1_000, "mille", "mila"),
        _ => italian_scaled(number, 1_000_000, " un milione", " milioni"),
    }
}

/// One scale of an Italian cardinal. `one` is how the scale reads with a multiplier of one
/// (`cento`, `mille`, `un milione`) and `many` how it reads with any other (`duecento`,
/// `duemila`, `due milioni`).
fn italian_scaled(number: u64, unit: u64, one: &str, many: &str) -> String {
    let count = number / unit;
    let head = if count == 1 {
        one.trim_start().to_owned()
    } else {
        format!("{}{many}", italian_cardinal(count))
    };
    match number % unit {
        0 => head,
        rest => format!("{head}{}", italian_cardinal(rest)),
    }
}

fn italian_ordinal(number: u64, feminine: bool) -> String {
    let ending = if feminine { 'a' } else { 'o' };
    if let Some(irregular) = ITALIAN_ORDINALS.get(number as usize).filter(|_| number > 0) {
        let mut stem = (*irregular).to_owned();
        stem.pop();
        stem.push(ending);
        return stem;
    }
    let cardinal = italian_cardinal(number);
    // `ventitré` keeps its vowel — `ventitreesimo` — where every other cardinal loses it.
    let stem = match cardinal.strip_suffix('\u{e9}') {
        Some(head) => format!("{head}e"),
        None => cardinal.trim_end_matches(['a', 'e', 'i', 'o']).to_owned(),
    };
    format!("{stem}esim{ending}")
}

fn scaled(number: u64, unit: u64, name: &str, ordinal: bool) -> String {
    let count = number / unit;
    let rest = number % unit;
    let head = format!("{} {name}", spell_magnitude(count, false));
    if rest == 0 {
        return if ordinal { format!("{head}th") } else { head };
    }
    format!("{head} {}", spell_magnitude(rest, ordinal))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The place marker a list of ordinary integers stands for.
    fn counts(values: &[i64]) -> Vec<Count> {
        values.iter().copied().map(Count::from).collect()
    }

    #[test]
    fn tokens_alternate() {
        assert_eq!(
            tokenize("1.1"),
            vec![
                Token::Format("1".to_owned()),
                Token::Separator(".".to_owned()),
                Token::Format("1".to_owned())
            ]
        );
        assert_eq!(
            tokenize("(a)"),
            vec![
                Token::Separator("(".to_owned()),
                Token::Format("a".to_owned()),
                Token::Separator(")".to_owned())
            ]
        );
    }

    #[test]
    fn decimal_tokens_pad_and_group() {
        assert_eq!(
            format_numbers(&counts(&[7]), "001", None, None, None, None, None),
            "007"
        );
        assert_eq!(
            format_numbers(
                &counts(&[1234567]),
                "1",
                None,
                Some(','),
                Some(3),
                None,
                None
            ),
            "1,234,567"
        );
    }

    #[test]
    fn alphabetic_and_roman_tokens() {
        assert_eq!(
            format_numbers(&counts(&[1]), "a", None, None, None, None, None),
            "a"
        );
        assert_eq!(
            format_numbers(&counts(&[27]), "a", None, None, None, None, None),
            "aa"
        );
        assert_eq!(
            format_numbers(&counts(&[4]), "i", None, None, None, None, None),
            "iv"
        );
        assert_eq!(
            format_numbers(&counts(&[1990]), "I", None, None, None, None, None),
            "MCMXC"
        );
    }

    #[test]
    fn multi_level_numbers_use_the_separators() {
        assert_eq!(
            format_numbers(&counts(&[1, 2, 3]), "1.1.1", None, None, None, None, None),
            "1.2.3"
        );
        assert_eq!(
            format_numbers(&counts(&[2, 3]), "(1)-a", None, None, None, None, None),
            "(2)-c"
        );
    }

    #[test]
    fn words_and_ordinals() {
        assert_eq!(
            format_numbers(&counts(&[21]), "w", None, None, None, None, None),
            "twenty-one"
        );
        assert_eq!(
            format_numbers(&counts(&[3]), "Ww", None, None, None, None, None),
            "Three"
        );
        assert_eq!(
            format_numbers(&counts(&[1]), "1", None, None, None, Some("yes"), None),
            "1st"
        );
        assert_eq!(
            format_numbers(&counts(&[112]), "1", None, None, None, Some("yes"), None),
            "112th"
        );
    }

    #[test]
    fn other_unicode_digit_families_are_used() {
        // Fullwidth digits, U+FF10 onwards.
        assert_eq!(
            format_numbers(&counts(&[42]), "\u{ff11}", None, None, None, None, None),
            "\u{ff14}\u{ff12}"
        );
    }

    #[test]
    fn numbering_sequences_are_recognized() {
        assert_eq!(
            format_numbers(&counts(&[3]), "\u{2460}", None, None, None, None, None),
            "\u{2462}"
        );
        assert_eq!(
            format_numbers(&counts(&[0]), "\u{2460}", None, None, None, None, None),
            "\u{24ea}"
        );
        assert_eq!(
            format_numbers(&counts(&[2]), "\u{2474}", None, None, None, None, None),
            "\u{2475}"
        );
    }

    #[test]
    fn an_empty_place_marker_formats_to_nothing() {
        assert_eq!(
            format_numbers(&counts(&[]), "1", None, None, None, None, None),
            ""
        );
    }

    #[test]
    fn german_numbers_are_spelled_and_declined() {
        // `number-0812`: the cardinals.
        for (value, expected) in [
            (3, "drei"),
            (10, "zehn"),
            (13, "dreizehn"),
            (20, "zwanzig"),
            (100, "einhundert"),
            (115, "einhundertf\u{fc}nfzehn"),
            (134, "einhundertvierunddrei\u{df}ig"),
            (200, "zweihundert"),
            (201, "zweihunderteins"),
            (210, "zweihundertzehn"),
            (230, "zweihundertdrei\u{df}ig"),
            (1000, "eintausend"),
            (1005, "eintausendf\u{fc}nf"),
            (
                2_134_816,
                "zwei Millionen einhundertvierunddrei\u{df}igtausendachthundertsechzehn",
            ),
        ] {
            assert_eq!(
                format_numbers(&counts(&[value]), "w", None, None, None, None, Some("de")),
                expected
            );
        }
        // `number-0813` and `number-0828`: the ordinals, whose ending the attribute names.
        for (value, spec, expected) in [
            (3, "-e", "dritte"),
            (10, "-er", "zehnter"),
            (13, "-es", "dreizehntes"),
            (20, "-en", "zwanzigsten"),
            (100, "-e", "einhundertste"),
            (115, "-er", "einhundertf\u{fc}nfzehnter"),
            (134, "-es", "einhundertvierunddrei\u{df}igstes"),
            (200, "-en", "zweihundertsten"),
            (201, "-e", "zweihunderterste"),
            (210, "-er", "zweihundertzehnter"),
            (230, "-es", "zweihundertdrei\u{df}igstes"),
            (1000, "-en", "eintausendsten"),
            (1005, "-e", "eintausendf\u{fc}nfte"),
            (3, "%spellout-ordinal", "dritte"),
            (10, "%spellout-ordinal", "zehnte"),
        ] {
            assert_eq!(
                format_numbers(
                    &counts(&[value]),
                    "w",
                    None,
                    None,
                    None,
                    Some(spec),
                    Some("de")
                ),
                expected
            );
        }
        // `number-0812` reads the largest of them under `format="Ww"`, where `Millionen` has to
        // stand apart as its own word for the title casing to reach the number after it.
        assert_eq!(
            format_numbers(
                &counts(&[2_134_816]),
                "Ww",
                None,
                None,
                None,
                None,
                Some("de")
            ),
            "Zwei Millionen Einhundertvierunddrei\u{df}igtausendachthundertsechzehn"
        );
        // `number-2506` alternates the language case by case, so English must be untouched.
        assert_eq!(
            format_numbers(&counts(&[1]), "Ww", None, None, None, None, Some("de")),
            "Eins"
        );
        assert_eq!(
            format_numbers(&counts(&[1]), "Ww", None, None, None, None, Some("en")),
            "One"
        );
    }

    #[test]
    fn italian_ordinals_take_the_gender_the_scheme_names() {
        let masculine = "%spellout-ordinal-masculine";
        let feminine = "%spellout-ordinal-feminine";
        let spell = |value: i64, spec: &str| {
            format_numbers(
                &counts(&[value]),
                "w",
                None,
                None,
                None,
                Some(spec),
                Some("it"),
            )
        };
        let ordinals = [
            "primo", "secondo", "terzo", "quarto", "quinto", "sesto", "settimo", "ottavo", "nono",
            "decimo",
        ];
        for (index, expected) in ordinals.iter().enumerate() {
            let value = index as i64 + 1;
            assert_eq!(spell(value, masculine), *expected);
            let mut female = (*expected).to_owned();
            female.pop();
            female.push('a');
            assert_eq!(spell(value, feminine), female);
        }
        // Past the irregular ten the cardinal loses its final vowel and takes `esimo`.
        assert_eq!(spell(11, masculine), "undicesimo");
        assert_eq!(spell(23, feminine), "ventitreesima");
        assert_eq!(
            format_numbers(&counts(&[21]), "w", None, None, None, None, Some("it")),
            "ventuno"
        );
    }

    #[test]
    fn a_greek_format_token_numbers_in_the_greek_alphabet() {
        // `number-0902`: α, β, γ … ω, then αα — under both `letter-value` spellings, since the
        // traditional Greek numerals `number-0901` names are discretionary.
        for letter_value in [None, Some("alphabetic"), Some("traditional")] {
            let spell = |value: i64| {
                format_numbers(
                    &counts(&[value]),
                    "\u{3b1}",
                    letter_value,
                    None,
                    None,
                    None,
                    None,
                )
            };
            assert_eq!(spell(1), "\u{3b1}");
            assert_eq!(spell(18), "\u{3c2}");
            assert_eq!(spell(19), "\u{3c3}");
            assert_eq!(spell(25), "\u{3c9}");
            assert_eq!(spell(26), "\u{3b1}\u{3b1}");
            assert_eq!(spell(27), "\u{3b1}\u{3b2}");
        }
        // Upper case has no final sigma, so its alphabet is one letter shorter.
        assert_eq!(
            format_numbers(&counts(&[24]), "\u{391}", None, None, None, None, None),
            "\u{3a9}"
        );
        // A token that is not a letter of any alphabet this processor knows keeps the decimal
        // fallback the specification allows.
        assert_eq!(
            format_numbers(&counts(&[3]), "\u{5d0}", None, None, None, None, None),
            "3"
        );
    }
}
