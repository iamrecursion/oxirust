//! A miniature CSS engine for `<style>` element content.
//!
//! SVG lets a `<style>` element hold a real (if small) CSS stylesheet: a
//! list of rules, each pairing a selector with a declaration block. This
//! module parses that stylesheet into [`Stylesheet`] and matches its
//! selectors against elements without ever touching a DOM: callers adapt
//! their own tree to [`ElementFacts`] and call [`Selector::matches_chain`].
//!
//! # The supported selector subset — a deliberate foundation cut
//!
//! This is not a general CSS selector engine (there already are excellent
//! ones; SVG stylesheets in practice use a narrow slice of CSS). Supported:
//!
//! * The universal selector, `*`.
//! * A type selector, `rect` (matched case-sensitively — SVG element names
//!   are XML local names, and XML names are case-sensitive).
//! * A class selector, `.a` (any number of classes may be chained).
//! * An id selector, `#b`.
//! * A compound selector combining the above with no space between them,
//!   in any order after the (optional, and if present, leading) type or
//!   universal part: `rect.a.b#c`, `.a#b`, `*.a`.
//! * The descendant combinator: white space between compound selectors,
//!   `g rect` (`rect` matches only if it has `g` among its ancestors,
//!   anywhere, not necessarily as its immediate parent).
//! * Grouping: a comma-separated list of selectors sharing one declaration
//!   block, `a, b { ... }`, split at parse time into one [`StyleRule`] per
//!   selector so each keeps its own [`Specificity`].
//!
//! Deliberately **not** supported: pseudo-classes and pseudo-elements
//! (`:hover`, `::first-line`), attribute selectors (`[foo="bar"]`), the
//! child and sibling combinators (`>`, `+`, `~`), and `@`-rules (`@media`,
//! `@import`, ...) other than skipping them wholesale. A selector that uses
//! any unsupported construct fails to parse; per the tolerant contract
//! below, that drops only the rule(s) using it.
//!
//! # Tolerant parsing
//!
//! [`Stylesheet::parse`] never fails. A rule whose selector (or, for a
//! grouped selector list, one member of the group) does not parse in the
//! supported subset above is skipped — the other selectors in its group,
//! and every other rule, are unaffected. An `@`-rule, block-form
//! (`@media (...) { ... }`) or semicolon-form (`@import url(...);`), is
//! skipped wholesale: neither its prelude nor its body (including any
//! ordinary rules nested inside a block `@`-rule) contributes anything. A
//! stray `}` with no matching `{` is treated as a recovery point and
//! skipped. Every declaration block is parsed with
//! [`crate::style::attr::parse_style_attribute`], so it inherits that
//! function's own per-declaration recovery.
//!
//! As in [`crate::style::attr`], the structural characters this grammar
//! depends on (`{`, `}`, `,`, `;`, and the `(...)`/comment/quote nesting
//! that can hide them) are always located by scanning the untouched input;
//! comments are recognized wherever `/* ... */` appears outside a quoted
//! string, and are stripped from selector text before it is parsed, but
//! never change how the input is otherwise segmented.
//!
//! Selector and class/id text is matched **case-sensitively** and is
//! restricted to ASCII letters, digits, hyphens, and underscores —
//! deliberately simpler than full CSS identifier syntax (which also allows
//! Unicode letters and backslash escapes), since SVG element, class, and id
//! names in practice do not need either.

use crate::style::attr::{Declaration, parse_style_attribute};
use crate::values::lex::is_wsp;
use std::fmt;

/// A CSS specificity: how strongly a selector's match should outweigh
/// another matching selector's, ordered `(id count, class count, type
/// count)` from most to least significant, matching the standard CSS
/// specificity comparison. The universal selector `*` contributes zero to
/// every component.
///
/// Deriving [`Ord`] on a 3-tuple compares the first field first, so
/// comparing two `Specificity` values directly implements CSS's
/// specificity-comparison rule: any difference in id count decides the
/// order outright; only when id counts are equal does class count decide
/// it; only when both are equal does type count decide it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Specificity(
    /// The number of id selectors (`#foo`) in the selector.
    pub u32,
    /// The number of class selectors (`.foo`) in the selector.
    pub u32,
    /// The number of type selectors (`foo`) in the selector; the universal
    /// selector `*` does not count.
    pub u32,
);

/// The DOM-independent facts about one element that [`Selector::matches_chain`]
/// needs, for one link of an ancestor-to-subject chain.
///
/// Callers adapt their own tree to this type; this module does not know
/// what a DOM node is.
#[derive(Debug, Clone, Copy)]
pub struct ElementFacts<'a> {
    /// The element's local (unprefixed) name, e.g. `"rect"`.
    pub name: &'a str,
    /// The element's `id` attribute value, if it has one.
    pub id: Option<&'a str>,
    /// The element's `class` attribute value, already split into
    /// individual tokens — see [`split_classes`].
    pub classes: &'a [&'a str],
}

/// One compound selector: an optional type (or universal) part, an
/// optional id, and any number of classes, all required to match at the
/// same element with no combinator between them (`rect.a.b#c`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompoundSelector {
    /// The required element local name, or [`None`] for the universal
    /// selector (and for a compound with no type part at all, e.g. `.a`,
    /// which CSS treats identically to `*.a`).
    pub element: Option<Box<str>>,
    /// The required `id` attribute value, if the compound has an id part.
    pub id: Option<Box<str>>,
    /// Every class the compound requires the element to have. Duplicates
    /// (`.a.a`) are harmless and simply checked twice.
    pub classes: Vec<Box<str>>,
}

impl CompoundSelector {
    /// Parses one compound selector token, such as `rect.a.b#c`, `.a`,
    /// `#b`, or `*`.
    ///
    /// `token` must contain no white space (callers split a descendant
    /// chain on white space first — see [`Selector::parse`]) and no
    /// grouping comma. Returns [`None`] if `token` is empty, uses a
    /// character outside the supported ASCII identifier/`.`/`#`/`*` set
    /// (which is how an unsupported construct such as `[foo]`, `:hover`,
    /// or a `>`/`+`/`~` combinator is rejected), repeats the type/universal
    /// part, or has a `.`/`#` with no name after it.
    #[must_use]
    pub fn parse(token: &str) -> Option<Self> {
        if token.is_empty() || !token.chars().all(is_compound_char) {
            return None;
        }
        let mut pos = 0usize;
        let mut element = None;
        let first = token[pos..].chars().next()?;
        if first == '*' {
            pos += 1;
        } else if is_ident_start(first) {
            let start = pos;
            pos += consume_ident_len(&token[pos..]);
            element = Some(Box::from(&token[start..pos]));
        }

        let mut id = None;
        let mut classes = Vec::new();
        while pos < token.len() {
            let marker = token[pos..].chars().next()?;
            pos += marker.len_utf8();
            let start = pos;
            let len = consume_ident_len(&token[pos..]);
            if len == 0 {
                return None;
            }
            pos += len;
            let name: Box<str> = Box::from(&token[start..pos]);
            match marker {
                '.' => classes.push(name),
                '#' => id = Some(name),
                _ => return None, // a second type/universal part, e.g. "a*b"
            }
        }

        Some(Self {
            element,
            id,
            classes,
        })
    }
}

/// True for the ASCII characters a compound selector may use: identifier
/// characters plus the `.`, `#`, and `*` structural markers. Any other
/// character — `>`, `+`, `~`, `[`, `]`, `(`, `)`, `:`, quotes, `@`, and so
/// on — means the token uses a construct this module does not support.
fn is_compound_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '#' | '*')
}

/// True for the first character of an identifier (element, class, or id
/// name): an ASCII letter, underscore, or hyphen.
fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || matches!(c, '-' | '_')
}

/// True for a non-first character of an identifier: an ASCII letter,
/// digit, underscore, or hyphen.
fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '_')
}

/// The byte length of the leading run of [`is_ident_continue`] characters
/// in `s`, or `0` if `s` does not start with one.
fn consume_ident_len(s: &str) -> usize {
    s.find(|c: char| !is_ident_continue(c)).unwrap_or(s.len())
}

impl CompoundSelector {
    /// Reports whether this compound selector matches an element with the
    /// given local name, `id` attribute, and (already-tokenized) class
    /// list. Matching is case-sensitive throughout, matching XML's own
    /// name and attribute-value case sensitivity.
    #[must_use]
    pub fn matches(&self, name: &str, id: Option<&str>, classes: &[&str]) -> bool {
        if let Some(element) = &self.element {
            if element.as_ref() != name {
                return false;
            }
        }
        if let Some(expected_id) = &self.id {
            if id != Some(expected_id.as_ref()) {
                return false;
            }
        }
        self.classes
            .iter()
            .all(|required| classes.iter().any(|actual| *actual == required.as_ref()))
    }
}

impl fmt::Display for CompoundSelector {
    /// Renders the compound selector back to CSS text, in a canonical
    /// `type?id?classes*` order (e.g. `rect#c.a.b`) that need not match the
    /// order the selector was originally written in (`.a#b.c` and
    /// `#b.a.c` both render the same way once parsed): this crate performs
    /// canonical re-emission rather than byte-exact round-tripping, the
    /// same policy [`crate::values::clock`] documents for clock values.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.element {
            Some(element) => write!(f, "{element}")?,
            None if self.id.is_none() && self.classes.is_empty() => write!(f, "*")?,
            None => {}
        }
        if let Some(id) = &self.id {
            write!(f, "#{id}")?;
        }
        for class in &self.classes {
            write!(f, ".{class}")?;
        }
        Ok(())
    }
}

/// A full selector: a chain of one or more [`CompoundSelector`]s joined by
/// the descendant combinator (white space), read left to right the way it
/// is written — `g rect` is `compounds: [g, rect]` — so the **last**
/// element of `compounds` is the subject the selector actually selects,
/// and every earlier compound is an ancestor constraint. A selector with
/// one compound (`rect`) has no ancestor constraint at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selector {
    /// The descendant chain, root-to-subject order; never empty for a
    /// selector [`Selector::parse`] returned.
    pub compounds: Vec<CompoundSelector>,
}

impl Selector {
    /// Parses one descendant-combinator chain, such as `g rect.a` — no
    /// grouping comma. Split a grouped selector list on top-level commas
    /// first (as [`Stylesheet::parse`] does) and parse each part
    /// separately.
    ///
    /// Returns [`None`] if `text` is empty (or white space only), or if any
    /// compound in the chain fails to parse — see
    /// [`CompoundSelector::parse`].
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        let mut compounds = Vec::new();
        for token in text.split_whitespace() {
            compounds.push(CompoundSelector::parse(token)?);
        }
        if compounds.is_empty() {
            return None;
        }
        Some(Self { compounds })
    }

    /// This selector's specificity: the number of id, class, and type
    /// selectors summed across every compound in the chain (the universal
    /// selector contributes nothing).
    #[must_use]
    pub fn specificity(&self) -> Specificity {
        let mut ids = 0u32;
        let mut classes = 0u32;
        let mut types = 0u32;
        for compound in &self.compounds {
            if compound.id.is_some() {
                ids += 1;
            }
            classes += u32::try_from(compound.classes.len()).unwrap_or(u32::MAX);
            if compound.element.is_some() {
                types += 1;
            }
        }
        Specificity(ids, classes, types)
    }

    /// Reports whether this selector matches `chain`, a root-first
    /// sequence of element facts ending with the subject element (the one
    /// being tested) as its **last** entry.
    ///
    /// The last compound of the selector must match the subject exactly;
    /// every earlier compound must match *some* ancestor further back in
    /// the chain, in order, but not necessarily the immediate parent —
    /// this module implements only the descendant combinator, never the
    /// child combinator. An empty chain never matches anything.
    ///
    /// ```
    /// use oxixml_svg::style::css::{ElementFacts, Selector};
    ///
    /// let selector = Selector::parse("g rect").expect("valid selector");
    /// let no_classes: &[&str] = &[];
    /// let chain = [
    ///     ElementFacts { name: "svg", id: None, classes: no_classes },
    ///     ElementFacts { name: "g", id: None, classes: no_classes },
    ///     ElementFacts { name: "a", id: None, classes: no_classes },
    ///     ElementFacts { name: "rect", id: None, classes: no_classes },
    /// ];
    /// assert!(selector.matches_chain(&chain));
    /// assert!(!selector.matches_chain(&chain[2..])); // no "g" ancestor left
    /// ```
    #[must_use]
    pub fn matches_chain(&self, chain: &[ElementFacts<'_>]) -> bool {
        let Some((subject, ancestors)) = chain.split_last() else {
            return false;
        };
        let Some((last_compound, earlier_compounds)) = self.compounds.split_last() else {
            return false;
        };
        if !last_compound.matches(subject.name, subject.id, subject.classes) {
            return false;
        }
        let mut remaining_ancestors = ancestors;
        for compound in earlier_compounds.iter().rev() {
            loop {
                let Some((candidate, rest)) = remaining_ancestors.split_last() else {
                    return false;
                };
                remaining_ancestors = rest;
                if compound.matches(candidate.name, candidate.id, candidate.classes) {
                    break;
                }
            }
        }
        true
    }
}

impl fmt::Display for Selector {
    /// Renders the descendant chain back to CSS text, space-separated, in
    /// the same left-to-right (root-to-subject) order it is stored in.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, compound) in self.compounds.iter().enumerate() {
            if index > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{compound}")?;
        }
        Ok(())
    }
}

/// One parsed rule: a selector paired with the declarations it sets when
/// that selector matches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyleRule {
    /// The selector that must match for `declarations` to apply.
    pub selector: Selector,
    /// The declarations this rule sets when `selector` matches, in source
    /// order.
    pub declarations: Vec<Declaration>,
}

/// A parsed `<style>` stylesheet: its rules, in source order, with grouped
/// selectors (`a, b { ... }`) already split into one [`StyleRule`] per
/// selector so each keeps its own [`Specificity`].
///
/// See the [module documentation](self) for exactly which selectors are
/// supported and how parse errors are tolerated.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stylesheet {
    /// The stylesheet's rules, in source order (and, within one grouped
    /// selector, in the order the group listed them).
    pub rules: Vec<StyleRule>,
}

/// Skips runs of white space and `/* ... */` comments starting at byte
/// offset `from` in `text`, returning the offset of the first character
/// that is neither. An unterminated trailing comment is skipped to the end
/// of `text`.
fn skip_trivia(text: &str, from: usize) -> usize {
    let mut pos = from;
    loop {
        let rest = &text[pos..];
        let trimmed = rest.trim_start_matches(is_wsp);
        pos += rest.len() - trimmed.len();
        if let Some(after_open) = trimmed.strip_prefix("/*") {
            pos += 2 + after_open.find("*/").map_or(after_open.len(), |i| i + 2);
            continue;
        }
        break;
    }
    pos
}

/// Scans `text` from byte offset `from` for the first occurrence of any
/// character in `targets` that appears at the *top level*: not inside a
/// `/* ... */` comment, not inside a `'...'`/`"..."` quoted string, and not
/// nested inside parentheses. Returns the match's absolute byte offset and
/// which target character it was.
///
/// This is the same notion of "top level" [`crate::style::attr`] scans
/// for, reused here to find the `{`/`}` that bound a rule and the `,` that
/// separates a grouped selector list — see the [module documentation](self).
fn find_top_level_any(text: &str, from: usize, targets: &[char]) -> Option<(usize, char)> {
    let mut iter = text[from..].char_indices().peekable();
    let mut quote: Option<char> = None;
    let mut in_comment = false;
    let mut paren_depth: u32 = 0;
    while let Some((rel, ch)) = iter.next() {
        if in_comment {
            if ch == '*' && iter.peek().map(|&(_, c)| c) == Some('/') {
                iter.next();
                in_comment = false;
            }
            continue;
        }
        if let Some(quote_char) = quote {
            if ch == '\\' {
                iter.next();
                continue;
            }
            if ch == quote_char {
                quote = None;
            }
            continue;
        }
        if ch == '/' && iter.peek().map(|&(_, c)| c) == Some('*') {
            iter.next();
            in_comment = true;
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            continue;
        }
        if ch == '(' {
            paren_depth += 1;
            continue;
        }
        if ch == ')' {
            paren_depth = paren_depth.saturating_sub(1);
            continue;
        }
        if paren_depth == 0 && targets.contains(&ch) {
            return Some((from + rel, ch));
        }
    }
    None
}

/// Splits `text` on every top-level occurrence of `target` (see
/// [`find_top_level_any`]), returning the pieces between separators. This
/// always yields at least one piece, even for an empty or separator-free
/// `text`.
fn split_top_level(text: &str, target: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    while let Some((offset, _)) = find_top_level_any(text, start, &[target]) {
        parts.push(&text[start..offset]);
        start = offset + target.len_utf8();
    }
    parts.push(&text[start..]);
    parts
}

/// Copies `text` with every `/* ... */` comment removed. Selector text
/// never legitimately contains a quoted string in the subset this module
/// supports, so unlike [`crate::style::attr`]'s comment stripper, this one
/// does not need to track quotes.
fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();
    while let Some((_, ch)) = chars.next() {
        if ch == '/' && chars.peek().map(|&(_, c)| c) == Some('*') {
            chars.next();
            let mut previous_was_star = false;
            for (_, comment_char) in chars.by_ref() {
                if previous_was_star && comment_char == '/' {
                    break;
                }
                previous_was_star = comment_char == '*';
            }
            continue;
        }
        out.push(ch);
    }
    out
}

/// Skips one `@`-rule starting at `text[at..]` (where `text[at..]` starts
/// with `@`), returning the byte offset just past it.
///
/// Both forms are handled: a semicolon-terminated at-rule
/// (`@import url(...);`) and a block at-rule (`@media (...) { ... }`),
/// whose body can itself contain further nested `{`/`}` pairs (ordinary
/// style rules, skipped along with everything else). A stray `}`
/// encountered before any `{` has been seen ends the at-rule there,
/// consistent with [`Stylesheet::parse`]'s general "recover at `}`" rule.
fn skip_at_rule(text: &str, at: usize) -> usize {
    let mut iter = text[at..].char_indices().peekable();
    let mut quote: Option<char> = None;
    let mut in_comment = false;
    let mut paren_depth: u32 = 0;
    let mut brace_depth: u32 = 0;
    let mut seen_brace = false;
    while let Some((rel, ch)) = iter.next() {
        if in_comment {
            if ch == '*' && iter.peek().map(|&(_, c)| c) == Some('/') {
                iter.next();
                in_comment = false;
            }
            continue;
        }
        if let Some(quote_char) = quote {
            if ch == '\\' {
                iter.next();
                continue;
            }
            if ch == quote_char {
                quote = None;
            }
            continue;
        }
        if ch == '/' && iter.peek().map(|&(_, c)| c) == Some('*') {
            iter.next();
            in_comment = true;
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            continue;
        }
        if ch == '(' {
            paren_depth += 1;
            continue;
        }
        if ch == ')' {
            paren_depth = paren_depth.saturating_sub(1);
            continue;
        }
        if paren_depth == 0 && !seen_brace && ch == ';' {
            return at + rel + ch.len_utf8();
        }
        if paren_depth == 0 && ch == '{' {
            seen_brace = true;
            brace_depth += 1;
            continue;
        }
        if paren_depth == 0 && ch == '}' {
            if brace_depth == 0 {
                return at + rel + ch.len_utf8();
            }
            brace_depth -= 1;
            if seen_brace && brace_depth == 0 {
                return at + rel + ch.len_utf8();
            }
            continue;
        }
    }
    text.len()
}

impl Stylesheet {
    /// Parses `text` (the character content of a `<style>` element) into a
    /// stylesheet, tolerating every parse failure the [module
    /// documentation](self) describes: this never fails and never panics.
    ///
    /// ```
    /// use oxixml_svg::style::css::Stylesheet;
    ///
    /// let sheet = Stylesheet::parse(
    ///     "@media (min-width: 1px) { rect { fill: red } } \
    ///      a, b.warn { stroke: black; stroke-width: 2 }",
    /// );
    /// // The `@media` block is skipped wholesale, including the rule
    /// // nested inside it; the grouped selector became two rules.
    /// assert_eq!(sheet.rules.len(), 2);
    /// assert_eq!(sheet.rules[0].selector.to_string(), "a");
    /// assert_eq!(sheet.rules[1].selector.to_string(), "b.warn");
    /// ```
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let mut rules = Vec::new();
        let mut pos = 0usize;
        loop {
            pos = skip_trivia(text, pos);
            if pos >= text.len() {
                break;
            }
            if text[pos..].starts_with('@') {
                pos = skip_at_rule(text, pos);
                continue;
            }
            match find_top_level_any(text, pos, &['{', '}']) {
                Some((brace, '{')) => {
                    let prelude = &text[pos..brace];
                    let body_start = brace + '{'.len_utf8();
                    let body_end = find_top_level_any(text, body_start, &['}'])
                        .map_or(text.len(), |(offset, _)| offset);
                    let declarations = parse_style_attribute(&text[body_start..body_end]);
                    for selector_text in split_top_level(prelude, ',') {
                        if let Some(selector) = Selector::parse(&strip_comments(selector_text)) {
                            rules.push(StyleRule {
                                selector,
                                declarations: declarations.clone(),
                            });
                        }
                    }
                    pos = if body_end < text.len() {
                        body_end + '}'.len_utf8()
                    } else {
                        text.len()
                    };
                }
                Some((stray, _)) => {
                    // A `}` with nothing open to close: discard whatever
                    // dangling prelude text preceded it and recover.
                    pos = stray + '}'.len_utf8();
                }
                None => break,
            }
        }
        Self { rules }
    }
}

/// Splits a `class` attribute value into its individual class tokens,
/// separated by runs of white space — the `class` attribute is always a
/// white-space-separated token list, never comma-separated.
///
/// ```
/// use oxixml_svg::style::css::split_classes;
///
/// let classes: Vec<&str> = split_classes("  warn  large\tcap ").collect();
/// assert_eq!(classes, vec!["warn", "large", "cap"]);
/// ```
pub fn split_classes(value: &str) -> impl Iterator<Item = &str> {
    value.split(is_wsp).filter(|token| !token.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_CLASSES: &[&str] = &[];

    fn facts<'a>(name: &'a str, id: Option<&'a str>, classes: &'a [&'a str]) -> ElementFacts<'a> {
        ElementFacts { name, id, classes }
    }

    #[test]
    fn universal_selector_matches_any_element() {
        let compound = CompoundSelector::parse("*").expect("universal selector parses");
        assert!(compound.matches("rect", None, NO_CLASSES));
        assert!(compound.matches("anything", Some("x"), &["a", "b"]));
    }

    #[test]
    fn type_selector_matches_only_that_element_case_sensitively() {
        let compound = CompoundSelector::parse("rect").expect("type selector parses");
        assert!(compound.matches("rect", None, NO_CLASSES));
        assert!(!compound.matches("circle", None, NO_CLASSES));
        assert!(
            !compound.matches("Rect", None, NO_CLASSES),
            "element names are matched case-sensitively"
        );
    }

    #[test]
    fn class_selector_matches_elements_that_have_that_class() {
        let compound = CompoundSelector::parse(".warn").expect("class selector parses");
        assert!(compound.matches("rect", None, &["warn"]));
        assert!(compound.matches("rect", None, &["big", "warn"]));
        assert!(!compound.matches("rect", None, &["big"]));
        assert!(!compound.matches("rect", None, NO_CLASSES));
    }

    #[test]
    fn id_selector_matches_only_that_id() {
        let compound = CompoundSelector::parse("#hero").expect("id selector parses");
        assert!(compound.matches("rect", Some("hero"), NO_CLASSES));
        assert!(!compound.matches("rect", Some("other"), NO_CLASSES));
        assert!(!compound.matches("rect", None, NO_CLASSES));
    }

    #[test]
    fn compound_selector_parses_every_part_regardless_of_order() {
        for token in ["rect.a.b#c", "rect#c.a.b", "#c.a.b", ".a#c.b"] {
            let compound = CompoundSelector::parse(token)
                .unwrap_or_else(|| panic!("{token:?} should parse as a compound selector"));
            assert_eq!(compound.id.as_deref(), Some("c"), "token {token:?}");
            let mut classes: Vec<&str> = compound.classes.iter().map(Box::as_ref).collect();
            classes.sort_unstable();
            assert_eq!(classes, vec!["a", "b"], "token {token:?}");
            if token.starts_with("rect") {
                assert_eq!(compound.element.as_deref(), Some("rect"));
            } else {
                assert_eq!(compound.element, None);
            }
        }
    }

    #[test]
    fn compound_selector_rejects_unsupported_constructs() {
        for token in [
            "rect:hover",
            "rect::first-line",
            "[title]",
            "rect[title]",
            "rect>circle",
            "rect+circle",
            "rect~circle",
            "rect(a)",
            "a*b",
            "**",
            "",
            ".",
            "#",
            "rect#",
        ] {
            assert_eq!(
                CompoundSelector::parse(token),
                None,
                "{token:?} uses a construct this crate does not support"
            );
        }
    }

    #[test]
    fn descendant_combinator_matches_any_ancestor_not_just_the_immediate_parent() {
        let selector = Selector::parse("g rect").expect("descendant selector parses");
        let chain = [
            facts("svg", None, NO_CLASSES),
            facts("g", None, NO_CLASSES),
            facts("a", None, NO_CLASSES),
            facts("rect", None, NO_CLASSES),
        ];
        assert!(
            selector.matches_chain(&chain),
            "rect has a `g` ancestor further up than its immediate parent"
        );
        assert!(
            !selector.matches_chain(&chain[2..]),
            "no `g` ancestor remains in [a, rect]"
        );
    }

    #[test]
    fn matches_chain_returns_false_for_an_empty_chain() {
        let selector = Selector::parse("*").expect("universal selector parses");
        assert!(!selector.matches_chain(&[]));
    }

    #[test]
    fn matches_chain_requires_the_subject_itself_to_match_the_last_compound() {
        let selector = Selector::parse("g rect").expect("descendant selector parses");
        let chain = [
            facts("svg", None, NO_CLASSES),
            facts("g", None, NO_CLASSES),
            facts("circle", None, NO_CLASSES),
        ];
        assert!(
            !selector.matches_chain(&chain),
            "the subject is a circle, not a rect, however good the ancestors are"
        );
    }

    #[test]
    fn a_selector_with_one_compound_ignores_ancestors_entirely() {
        let selector = Selector::parse("rect").expect("single-compound selector parses");
        let chain = [
            facts("svg", None, NO_CLASSES),
            facts("rect", None, NO_CLASSES),
        ];
        assert!(selector.matches_chain(&chain));
        assert!(selector.matches_chain(&chain[1..]));
    }

    #[test]
    fn grouped_selectors_split_into_separate_rules_sharing_declarations() {
        let sheet = Stylesheet::parse("a, b.warn { fill: red }");
        assert_eq!(sheet.rules.len(), 2);
        assert_eq!(sheet.rules[0].selector.to_string(), "a");
        assert_eq!(sheet.rules[1].selector.to_string(), "b.warn");
        assert_eq!(sheet.rules[0].declarations, sheet.rules[1].declarations);
        assert_eq!(sheet.rules[0].declarations.len(), 1);
        assert_eq!(&*sheet.rules[0].declarations[0].name, "fill");
    }

    #[test]
    fn specificity_orders_id_above_class_above_type() {
        let id_selector = Selector::parse("#a").expect("id selector parses");
        let many_classes = Selector::parse(".a.b.c.d.e").expect("class selector parses");
        let type_selector = Selector::parse("rect").expect("type selector parses");
        assert!(
            id_selector.specificity() > many_classes.specificity(),
            "one id always outweighs any number of classes"
        );
        assert!(
            many_classes.specificity() > type_selector.specificity(),
            "one class always outweighs any number of types"
        );
    }

    #[test]
    fn universal_selector_contributes_zero_specificity() {
        let universal = Selector::parse("*").expect("universal selector parses");
        assert_eq!(universal.specificity(), Specificity(0, 0, 0));
        let universal_with_class = Selector::parse("*.a").expect("selector parses");
        assert_eq!(universal_with_class.specificity(), Specificity(0, 1, 0));
    }

    #[test]
    fn descendant_chain_specificity_sums_every_compound() {
        let selector = Selector::parse("g.a rect#b.c.d").expect("selector parses");
        assert_eq!(selector.specificity(), Specificity(1, 3, 2));
    }

    #[test]
    fn stylesheet_parse_skips_block_at_rules_including_their_nested_rules() {
        let sheet = Stylesheet::parse(
            "@media (min-width: 1px) { rect { fill: red } } circle { fill: blue }",
        );
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector.to_string(), "circle");
    }

    #[test]
    fn stylesheet_parse_skips_semicolon_at_rules() {
        let sheet = Stylesheet::parse("@import url(foo.css); rect { fill: red }");
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector.to_string(), "rect");
    }

    #[test]
    fn stylesheet_parse_tolerates_a_quoted_semicolon_inside_an_at_rule() {
        let sheet = Stylesheet::parse(r#"@import "foo;bar.css"; rect { fill: red }"#);
        assert_eq!(
            sheet.rules.len(),
            1,
            "the `;` inside the quoted url must not end the at-rule early"
        );
        assert_eq!(sheet.rules[0].selector.to_string(), "rect");
    }

    #[test]
    fn stylesheet_parse_recovers_at_a_stray_closing_brace() {
        let sheet = Stylesheet::parse("} rect { fill: red }");
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector.to_string(), "rect");
    }

    #[test]
    fn stylesheet_parse_tolerates_comments_between_and_within_rules() {
        let sheet = Stylesheet::parse("/* c1 */ rect /* c2 */ { fill: red } /* c3 */");
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector.to_string(), "rect");
        assert_eq!(&*sheet.rules[0].declarations[0].value, "red");
    }

    #[test]
    fn stylesheet_parse_drops_only_the_rule_using_an_unsupported_selector() {
        let sheet = Stylesheet::parse("rect > circle { fill: red } path { fill: blue }");
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector.to_string(), "path");
    }

    #[test]
    fn stylesheet_parse_handles_an_empty_declaration_block() {
        let sheet = Stylesheet::parse("rect {}");
        assert_eq!(sheet.rules.len(), 1);
        assert!(sheet.rules[0].declarations.is_empty());
    }

    #[test]
    fn stylesheet_parse_reuses_the_style_attribute_parsers_own_recovery() {
        let sheet = Stylesheet::parse("rect { fill:red; not a declaration; stroke:blue }");
        assert_eq!(sheet.rules.len(), 1);
        let names: Vec<&str> = sheet.rules[0]
            .declarations
            .iter()
            .map(|declaration| &*declaration.name)
            .collect();
        assert_eq!(names, vec!["fill", "stroke"]);
    }

    #[test]
    fn stylesheet_default_and_empty_input_both_produce_no_rules() {
        assert_eq!(Stylesheet::default().rules, Vec::new());
        assert_eq!(Stylesheet::parse("").rules, Vec::new());
        assert_eq!(
            Stylesheet::parse("   /* only a comment */  ").rules,
            Vec::new()
        );
    }

    #[test]
    fn split_classes_splits_on_any_run_of_white_space() {
        assert_eq!(
            split_classes("  warn  large\tcap \n").collect::<Vec<_>>(),
            vec!["warn", "large", "cap"]
        );
        assert_eq!(split_classes("").collect::<Vec<_>>(), Vec::<&str>::new());
        assert_eq!(split_classes("solo").collect::<Vec<_>>(), vec!["solo"]);
    }

    #[test]
    fn compound_selector_display_uses_a_canonical_type_id_class_order() {
        let compound = CompoundSelector::parse(".a#b.c").expect("compound selector parses");
        assert_eq!(compound.to_string(), "#b.a.c");
    }

    #[test]
    fn selector_display_joins_compounds_with_a_single_space() {
        let selector = Selector::parse("g   rect.a").expect("selector parses");
        assert_eq!(selector.to_string(), "g rect.a");
    }
}
