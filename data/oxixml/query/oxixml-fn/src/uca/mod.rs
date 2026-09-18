//! The Unicode Collation Algorithm collation URIs of
//! [XSLT 3.0 §5.3.4](https://www.w3.org/TR/xslt-30/#uca-collation-uri), which
//! [F&O 3.1 §5.3.5](https://www.w3.org/TR/xpath-functions-31/#uca-collations) adopts for every
//! collation-sensitive function.
//!
//! A URI of the form
//! `http://www.w3.org/2013/collation/UCA?lang=de;strength=primary;caseFirst=upper` names a
//! comparison by *levels*: base letters first, then accents, then case, then the punctuation and
//! whitespace that the `alternate` parameter may have set aside, and finally the code points
//! themselves.
//!
//! # What this implements
//!
//! The whole level structure of [UTS #10](https://www.unicode.org/reports/tr10/) over the root
//! weight table described below, with the parameters `lang`, `strength`, `alternate`,
//! `maxVariable`, `backwards`, `caseLevel`, `caseFirst`, `numeric`, `normalization` and `fallback`
//! all honoured:
//!
//! * **primary** — the base characters, case- and accent-blind. A character that the root table
//!   weighs as several — `ß` as `ss` — contributes each of them, so `Straße` and `Strasse` are one
//!   word at this level;
//! * **secondary** — the combining marks the canonical decomposition produced, compared left to
//!   right, or right to left under `backwards=yes`;
//! * **case level** — under `caseLevel=yes`, case on its own, before the tertiary level and
//!   whatever the strength is;
//! * **tertiary** — case, and then the compatibility variants: the `ss` of `ß` differs here from
//!   the `ss` of `Strasse`, which is what makes the two words distinguishable again;
//! * **quaternary** — the variable characters, which `alternate=shifted` defers to this level and
//!   `alternate=blanked` discards altogether;
//! * **identical** — the code points of the canonical decomposition.
//!
//! # The repertoire
//!
//! The DUCET is some thirty thousand entries; this crate is self-contained and carries the part an
//! XML processor needs, computing the rest. **Explicit** weights are given to:
//!
//! * the **variable** characters — whitespace, then punctuation, then the general symbols, then the
//!   currency symbols, which is both the root order and the order the four `maxVariable` groups
//!   nest in, and which places all of them below the digits and the letters;
//! * the **decimal digits** of every script, ordered by value rather than by code point, so that
//!   `٣` weighs as `3` does;
//! * the **basic Latin letters** `a`–`z`, which every accented Latin letter reduces to once the
//!   canonical decomposition has moved its accents into combining marks;
//! * the few Latin characters that weigh as *several* letters and that no canonical decomposition
//!   produces: `ß` and `ẞ` as `ss`, `ĳ` as `ij`, and the `ﬀ`, `ﬁ`, `ﬂ`, `ﬃ`, `ﬄ`, `ﬅ` and `ﬆ`
//!   ligatures as their parts.
//!
//! Every other code point takes an **implicit** weight, which is [UTS
//! #10 §10](https://www.unicode.org/reports/tr10/#Implicit_Weights)'s answer for a character the
//! table does not name: a weight derived from the code point, above every explicit one. So the
//! scripts the table does not enumerate — Greek, Cyrillic, Hebrew, Arabic, the Indic scripts, the
//! kana and the Han ideographs — sort after the Latin letters, and among themselves in Unicode's
//! own block order, which is the order the root table puts them in; within a script, by code
//! point. What that costs against the whole DUCET is the relative placement of two unenumerated
//! scripts whose blocks and root weights disagree, and the accent orderings a script gives its own
//! marks. Neither affects the Latin, digit and punctuation text an XML processor compares.
//!
//! # Language tailorings
//!
//! `lang=en` and `lang=de` read the root order unchanged — German's DIN 5007-1 ordering *is* the
//! root order, since `ä` sorts as `a` with a secondary difference and `ß` as `ss`. `lang=sv` places
//! `å`, `ä` and `ö` after `z`, sorts `æ` with `ä`, `ø` with `ö` and `ü` with `y`, as SS 613621
//! prescribes. A `lang` with no tailoring here is answered with the root order under the default
//! `fallback=yes`, and with `FOCH0002` under `fallback=no` — never with a silent substitution.
//! [`UcaCollation::is_fallback`] reports which of the two happened.
//!
//! # Collation units and expansions
//!
//! [`Collation::collation_units`] must map each input character to exactly one unit, so that
//! `fn:substring-before` can report an offset in the original string. An *expansion* cannot be
//! written that way: `ß` is one character and two collation elements. So the five substring
//! functions see `ß` as a single unit that matches only itself, and `fn:contains("Straße", "ss")`
//! is false where `fn:compare("Straße", "strasse")` at primary strength is zero. The alternative —
//! refusing to supply collation units at all, and raising `FOCH0004` from all five functions — is
//! the worse answer for every string that contains no expansion, which is nearly all of them.

mod tailoring;
mod weights;

use crate::collation::Collation;
use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_unicode::case::simple_fold;
use oxixml_unicode::norm::nfd;
use std::cmp::Ordering;
use tailoring::Tailoring;

/// The URI every UCA collation name starts with.
pub const UCA_COLLATION_BASE: &str = "http://www.w3.org/2013/collation/UCA";

/// The comparison levels a UCA collation may run to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Strength {
    /// Base characters only.
    Primary,
    /// Base characters and accents.
    Secondary,
    /// Base characters, accents and case.
    Tertiary,
    /// Also the variable characters `alternate=shifted` set aside.
    Quaternary,
    /// Also the code points themselves.
    Identical,
}

/// How the *variable* characters — whitespace and punctuation — take part.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alternate {
    /// They are ordinary characters, weighed at the primary level. The default.
    NonIgnorable,
    /// They are ignored at every level.
    Blanked,
    /// They are ignored at the first three levels and compared at the fourth.
    Shifted,
}

/// Which characters count as variable, from the least to the most inclusive.
///
/// The four *reordering groups* of [UTS #35
/// §3.6](https://www.unicode.org/reports/tr35/tr35-collation.html#Common_Settings) nest in the
/// order `space` ⊂ `punct` ⊂ `symbol` ⊂ `currency`: currency symbols sort *after* the general
/// symbols in the root order, so naming `symbol` leaves currency alone while naming `currency`
/// takes both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MaxVariable {
    /// Whitespace only.
    Space,
    /// Whitespace and punctuation. The default.
    Punct,
    /// Whitespace, punctuation and the general symbols — but not currency.
    Symbol,
    /// Whitespace, punctuation and every symbol, currency included.
    Currency,
}

impl MaxVariable {
    /// The highest root group this setting makes variable.
    const fn group(self) -> u64 {
        match self {
            Self::Space => weights::GROUP_WHITESPACE,
            Self::Punct => weights::GROUP_PUNCTUATION,
            Self::Symbol => weights::GROUP_SYMBOL,
            Self::Currency => weights::GROUP_CURRENCY,
        }
    }
}

/// A collation named by a `http://www.w3.org/2013/collation/UCA` URI.
#[derive(Debug, Clone)]
pub struct UcaCollation {
    uri: String,
    language: Option<String>,
    tailoring: Tailoring,
    strength: Strength,
    alternate: Alternate,
    max_variable: MaxVariable,
    backwards: bool,
    case_level: bool,
    upper_first: bool,
    numeric: bool,
}

/// One collation element: what a character contributes to each level of the sort key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Element {
    /// The base character's weight, or `0` for a combining mark, which has none.
    primary: u64,
    /// The accent: [`weights::SECONDARY_COMMON`] for a plain base character, the mark's own weight
    /// for a combining mark.
    secondary: u64,
    /// `0` for lower case and caseless characters, `1` for upper case.
    case: u8,
    /// `0` for an ordinary character, higher for a compatibility variant such as either `s` of `ß`.
    variant: u8,
    /// Whether this element is a variable character.
    variable: bool,
}

impl Element {
    /// An element with a primary weight and no accent — a digit, or the length of a digit run.
    const fn plain(primary: u64) -> Self {
        Self {
            primary,
            secondary: weights::SECONDARY_COMMON,
            case: 0,
            variant: weights::VARIANT_NONE,
            variable: false,
        }
    }

    /// A combining mark, which weighs at the secondary level only.
    fn mark(character: char) -> Self {
        Self {
            primary: 0,
            secondary: weights::mark_secondary(character),
            case: 0,
            variant: weights::VARIANT_NONE,
            variable: false,
        }
    }
}

/// An element once the `alternate` parameter has had its say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Weighed {
    primary: u64,
    secondary: u64,
    case: u8,
    variant: u8,
    /// The fourth-level weight, which only `alternate=shifted` gives out.
    quaternary: u64,
    /// Whether variable weighting took the element out of the first three levels.
    ignored: bool,
}

impl UcaCollation {
    /// Reads a UCA collation URI.
    ///
    /// Returns `Ok(None)` when `uri` does not name a UCA collation at all, so that a registry can
    /// go on looking.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Foch0002`] when the URI *is* a UCA URI but carries a parameter, or a
    /// parameter value, that the specification does not define — or, under `fallback=no`, one this
    /// collation cannot honour exactly.
    pub fn parse(uri: &str) -> FnResult<Option<Self>> {
        let Some(rest) = uri.strip_prefix(UCA_COLLATION_BASE) else {
            return Ok(None);
        };
        let mut collation = Self {
            uri: uri.to_owned(),
            language: None,
            tailoring: Tailoring::Root,
            strength: Strength::Tertiary,
            alternate: Alternate::NonIgnorable,
            max_variable: MaxVariable::Punct,
            backwards: false,
            case_level: false,
            upper_first: false,
            numeric: false,
        };
        let parameters = match rest {
            "" => "",
            rest => rest.strip_prefix('?').ok_or_else(|| unsupported(uri))?,
        };
        // XSLT 3.0 §5.3.4: `fallback=no` says the caller would rather have `FOCH0002` than an
        // approximation, and `fallback=yes` — the default — says a parameter this implementation
        // cannot honour, or does not recognise at all, is to be *ignored*. The flag governs
        // every other parameter, so it is read first however late in the URI it appears.
        let mut fallback = true;
        for parameter in parameters.split(';').filter(|part| !part.is_empty()) {
            if let Some(("fallback", value)) = parameter.split_once('=') {
                fallback = yes_no(uri, value)?;
            }
        }
        // A parameter this implementation cannot honour exactly: reported only under
        // `fallback=no`, and otherwise left at the default.
        let unhonoured = |()| -> FnResult<()> {
            if fallback {
                Ok(())
            } else {
                Err(unsupported(uri))
            }
        };
        for parameter in parameters.split(';').filter(|part| !part.is_empty()) {
            let Some((key, value)) = parameter.split_once('=') else {
                unhonoured(())?;
                continue;
            };
            match key {
                "fallback" => {}
                "lang" => collation.language = Some(value.to_owned()),
                // The DUCET version and the reordering groups both need weights beyond the root
                // table this collation carries, so neither can be honoured exactly.
                "version" | "reorder" => unhonoured(())?,
                "strength" => {
                    collation.strength = match value {
                        "primary" | "1" => Strength::Primary,
                        "secondary" | "2" => Strength::Secondary,
                        "tertiary" | "3" => Strength::Tertiary,
                        "quaternary" | "4" => Strength::Quaternary,
                        "identical" | "5" => Strength::Identical,
                        _ => {
                            unhonoured(())?;
                            collation.strength
                        }
                    };
                }
                "alternate" => {
                    collation.alternate = match value {
                        "non-ignorable" => Alternate::NonIgnorable,
                        "blanked" => Alternate::Blanked,
                        "shifted" => Alternate::Shifted,
                        _ => {
                            unhonoured(())?;
                            collation.alternate
                        }
                    };
                }
                "maxVariable" => {
                    collation.max_variable = match value {
                        "space" => MaxVariable::Space,
                        "punct" => MaxVariable::Punct,
                        "currency" => MaxVariable::Currency,
                        "symbol" => MaxVariable::Symbol,
                        _ => {
                            unhonoured(())?;
                            collation.max_variable
                        }
                    };
                }
                "backwards" => collation.backwards = optional_yes_no(uri, value, fallback)?,
                "normalization" => {
                    // This collation always decomposes, so `normalization=yes` is the behaviour
                    // either way and `normalization=no` costs nothing to honour as `yes`.
                    optional_yes_no(uri, value, fallback)?;
                }
                "caseLevel" => collation.case_level = optional_yes_no(uri, value, fallback)?,
                "caseFirst" => {
                    collation.upper_first = match value {
                        "upper" => true,
                        "lower" => false,
                        _ => {
                            unhonoured(())?;
                            collation.upper_first
                        }
                    };
                }
                "numeric" => collation.numeric = optional_yes_no(uri, value, fallback)?,
                _ => unhonoured(())?,
            }
        }
        // A `lang` this collation has no tailoring for is answered with the root order, which
        // `fallback=no` says the caller would rather have as `FOCH0002`.
        if let Some(language) = collation.language.as_deref() {
            match Tailoring::for_tag(language) {
                Some(tailoring) => collation.tailoring = tailoring,
                None if fallback => {}
                None => return Err(unsupported(uri)),
            }
        }
        Ok(Some(collation))
    }

    /// Whether the URI asked for a language tailoring this collation does not have, and is
    /// answered with the root ordering instead.
    ///
    /// Only a URI with `fallback=yes` — the default — can be in that state: `fallback=no` makes
    /// [`UcaCollation::parse`] raise `FOCH0002` rather than build a collation that does not order
    /// the way its `lang` asked for. A URI with no `lang` at all asked for no tailoring and so is
    /// not a fallback for anything.
    #[must_use]
    pub fn is_fallback(&self) -> bool {
        self.language
            .as_deref()
            .is_some_and(|tag| Tailoring::for_tag(tag).is_none())
    }

    /// Whether `character` is one the `alternate` parameter may set aside.
    fn is_variable(&self, character: char) -> bool {
        weights::variable_group(character).is_some_and(|group| group <= self.max_variable.group())
    }

    /// The collation elements of a string, one or more per base character of its canonical
    /// decomposition, with a run of digits collapsed under `numeric=yes`.
    fn elements(&self, value: &str) -> Vec<Element> {
        let decomposed = nfd(value);
        let characters: Vec<char> = decomposed.chars().collect();
        let mut elements = Vec::with_capacity(characters.len());
        let mut index = 0;
        while index < characters.len() {
            let base = characters[index];
            if weights::is_combining(base) {
                // A mark with no base character of its own still weighs at the secondary level.
                elements.push(Element::mark(base));
                index += 1;
                continue;
            }
            if self.numeric {
                if let Some(value) = weights::decimal_digit(base) {
                    let mut digits = vec![value];
                    index += 1;
                    while let Some(next) = characters
                        .get(index)
                        .copied()
                        .and_then(weights::decimal_digit)
                    {
                        digits.push(next);
                        index += 1;
                    }
                    push_number(&digits, &mut elements);
                    continue;
                }
            }
            index += 1;
            // The marks that follow the base character, which a tailoring may absorb into a letter
            // of its own — Swedish reads `a` and a ring above as `å`, the letter after `z`.
            let marks_start = index;
            while characters
                .get(index)
                .is_some_and(|character| weights::is_combining(*character))
            {
                index += 1;
            }
            let marks = &characters[marks_start..index];
            let absorbed = self.push_base(base, marks, &mut elements);
            for mark in marks.iter().skip(absorbed) {
                elements.push(Element::mark(*mark));
            }
        }
        elements
    }

    /// Weighs one base character and answers how many of the marks that follow it a tailoring took
    /// into the letter.
    fn push_base(&self, base: char, marks: &[char], elements: &mut Vec<Element>) -> usize {
        if let Some(letter) = self.tailoring.letter(base, marks) {
            elements.push(Element {
                primary: letter.primary,
                secondary: letter.secondary,
                case: weights::case_of(base),
                variant: weights::VARIANT_NONE,
                variable: false,
            });
            return letter.marks_used;
        }
        match weights::expansion(base) {
            // The replacements are themselves plain letters, so one round of expansion is enough.
            Some((letters, variant)) => {
                for part in letters.chars() {
                    if let Some((primary, secondary)) = weights::plain_weight(part) {
                        elements.push(Element {
                            primary,
                            secondary,
                            case: weights::case_of(part),
                            variant,
                            variable: false,
                        });
                    }
                }
            }
            None => {
                if let Some((primary, secondary)) = weights::plain_weight(base) {
                    elements.push(Element {
                        primary,
                        secondary,
                        case: weights::case_of(base),
                        variant: weights::VARIANT_NONE,
                        variable: self.is_variable(base),
                    });
                }
            }
        }
        0
    }

    /// Applies the variable weighting of [UTS
    /// #10 §3.4](https://www.unicode.org/reports/tr10/#Variable_Weighting).
    fn weigh(&self, elements: &[Element]) -> Vec<Weighed> {
        let mut weighed = Vec::with_capacity(elements.len());
        // An ignorable element that *follows* a variable one is ignored with it, so that the
        // accents of a set-aside punctuation mark do not outlive the mark itself.
        let mut after_variable = false;
        for element in elements {
            let variable = element.variable && self.alternate != Alternate::NonIgnorable;
            let trailing = element.primary == 0 && after_variable;
            if element.primary != 0 {
                after_variable = variable;
            }
            let ignored = variable || trailing;
            let quaternary = match (self.alternate, ignored) {
                // A variable element weighs its own primary at the fourth level; every other one
                // weighs the maximum, which is what makes `a-bc` sort before `abc`.
                (Alternate::Shifted, true) => element.primary,
                (Alternate::Shifted, false) => weights::QUATERNARY_MAX,
                _ => 0,
            };
            weighed.push(Weighed {
                primary: if ignored { 0 } else { element.primary },
                secondary: if ignored { 0 } else { element.secondary },
                case: element.case,
                variant: element.variant,
                quaternary,
                ignored,
            });
        }
        weighed
    }

    /// The second level: every non-zero secondary weight, reversed under `backwards=yes`.
    fn secondaries(&self, weighed: &[Weighed]) -> Vec<u64> {
        let mut level: Vec<u64> = weighed
            .iter()
            .filter(|element| element.secondary != 0)
            .map(|element| element.secondary)
            .collect();
        if self.backwards {
            level.reverse();
        }
        level
    }

    /// The case level: the case of every element that has a primary weight.
    ///
    /// Filtering on the primary is what keeps the level to the *letters*: a combining mark is
    /// caseless and would otherwise make `DÃTABASE` differ from `DATABASE` at a level that is
    /// supposed to see nothing but case.
    fn cases(&self, weighed: &[Weighed]) -> Vec<u8> {
        weighed
            .iter()
            .filter(|element| element.primary != 0)
            .map(|element| self.case_rank(element))
            .collect()
    }

    /// The third level: case, and then the compatibility variants.
    fn tertiaries(&self, weighed: &[Weighed]) -> Vec<u16> {
        weighed
            .iter()
            .filter(|element| !element.ignored)
            .map(|element| weights::tertiary_weight(self.case_rank(element), element.variant))
            .collect()
    }

    /// `0` for the case that sorts first and `1` for the other, which is what `caseFirst` decides.
    fn case_rank(&self, element: &Weighed) -> u8 {
        let case = u8::from(element.case != 0);
        if self.upper_first { 1 - case } else { case }
    }
}

/// The first level: every non-zero primary weight.
fn primaries(weighed: &[Weighed]) -> Vec<u64> {
    weighed
        .iter()
        .filter(|element| element.primary != 0)
        .map(|element| element.primary)
        .collect()
}

/// The fourth level, which only `alternate=shifted` fills in.
fn quaternaries(weighed: &[Weighed]) -> Vec<u64> {
    weighed
        .iter()
        .filter(|element| element.quaternary != 0)
        .map(|element| element.quaternary)
        .collect()
}

fn yes_no(uri: &str, value: &str) -> FnResult<bool> {
    match value {
        "yes" => Ok(true),
        "no" => Ok(false),
        _ => Err(unsupported(uri)),
    }
}

/// [`yes_no`] under the `fallback` rule: a value that is neither `yes` nor `no` is `FOCH0002`
/// only when `fallback=no`, and otherwise leaves the parameter at its default (`false`).
fn optional_yes_no(uri: &str, value: &str, fallback: bool) -> FnResult<bool> {
    match yes_no(uri, value) {
        Ok(flag) => Ok(flag),
        Err(_) if fallback => Ok(false),
        Err(error) => Err(error),
    }
}

fn unsupported(uri: &str) -> FnError {
    FnError::new(
        FnErrorCode::Foch0002,
        format!("the collation `{uri}` is not supported"),
    )
}

/// Weighs a run of decimal digit *values* as one number, which is what `numeric=yes` asks for.
fn push_number(digits: &[u64], elements: &mut Vec<Element>) {
    // The run weighs as its *length* and then its digits, so that `2` sorts before `10` however
    // many digits either has — and so that a number wider than any machine integer still orders by
    // magnitude. Leading zeros count towards neither, so `007` and `7` are one number.
    let start = digits
        .iter()
        .position(|digit| *digit != 0)
        .unwrap_or(digits.len());
    let significant = &digits[start..];
    let length = u64::try_from(significant.len()).unwrap_or(u64::MAX);
    elements.push(Element::plain(weights::number_length_primary(length)));
    for value in significant {
        elements.push(Element::plain(weights::digit_primary(*value)));
    }
}

/// The byte that separates two levels of a sort key, below every byte a weight can be written as.
const LEVEL_SEPARATOR: u8 = 0x00;

/// Writes `weight` into a sort key as `width` base-255 digits, most significant first.
///
/// Every digit is written one above its value, so that no byte of a weight can be the
/// [`LEVEL_SEPARATOR`]. A level that is a prefix of another then compares below it, which is
/// exactly what comparing the two levels as sequences of weights does.
fn push_weight(key: &mut Vec<u8>, weight: u64, width: usize) {
    let mut digits = [0_u8; weights::MAX_WEIGHT_WIDTH];
    let mut rest = weight;
    for slot in digits.iter_mut().take(width).rev() {
        *slot = u8::try_from(rest % 255).unwrap_or(0).saturating_add(1);
        rest /= 255;
    }
    key.extend_from_slice(&digits[..width.min(digits.len())]);
}

impl Collation for UcaCollation {
    fn uri(&self) -> &str {
        &self.uri
    }

    fn compare(&self, left: &str, right: &str) -> Ordering {
        let a = self.weigh(&self.elements(left));
        let b = self.weigh(&self.elements(right));

        let order = primaries(&a).cmp(&primaries(&b));
        if order != Ordering::Equal {
            return order;
        }
        if self.strength >= Strength::Secondary {
            let order = self.secondaries(&a).cmp(&self.secondaries(&b));
            if order != Ordering::Equal {
                return order;
            }
        }
        if self.case_level {
            let order = self.cases(&a).cmp(&self.cases(&b));
            if order != Ordering::Equal {
                return order;
            }
        }
        if self.strength >= Strength::Tertiary {
            let order = self.tertiaries(&a).cmp(&self.tertiaries(&b));
            if order != Ordering::Equal {
                return order;
            }
        }
        if self.strength >= Strength::Quaternary && self.alternate == Alternate::Shifted {
            let order = quaternaries(&a).cmp(&quaternaries(&b));
            if order != Ordering::Equal {
                return order;
            }
        }
        if self.strength >= Strength::Identical {
            return nfd(left).cmp(&nfd(right));
        }
        Ordering::Equal
    }

    fn sort_key(&self, value: &str) -> Option<Vec<u8>> {
        // UTS #10 §7: a sort key is the levels laid end to end, each terminated by a separator
        // lower than any byte a weight can be written as, so that comparing two keys byte by byte
        // gives the same answer as comparing them level by level. The levels are built by the same
        // methods `compare` uses, so the two cannot drift apart.
        let weighed = self.weigh(&self.elements(value));
        let mut key = Vec::with_capacity(weighed.len() * weights::PRIMARY_WIDTH);
        for weight in primaries(&weighed) {
            push_weight(&mut key, weight, weights::PRIMARY_WIDTH);
        }
        if self.strength >= Strength::Secondary {
            key.push(LEVEL_SEPARATOR);
            for weight in self.secondaries(&weighed) {
                push_weight(&mut key, weight, weights::SECONDARY_WIDTH);
            }
        }
        if self.case_level {
            key.push(LEVEL_SEPARATOR);
            for rank in self.cases(&weighed) {
                push_weight(&mut key, u64::from(rank), weights::CASE_WIDTH);
            }
        }
        if self.strength >= Strength::Tertiary {
            key.push(LEVEL_SEPARATOR);
            for weight in self.tertiaries(&weighed) {
                push_weight(&mut key, u64::from(weight), weights::TERTIARY_WIDTH);
            }
        }
        if self.strength >= Strength::Quaternary && self.alternate == Alternate::Shifted {
            key.push(LEVEL_SEPARATOR);
            for weight in quaternaries(&weighed) {
                push_weight(&mut key, weight, weights::PRIMARY_WIDTH);
            }
        }
        if self.strength >= Strength::Identical {
            key.push(LEVEL_SEPARATOR);
            // No UTF-8 byte of an XML character is the separator, so the level stays readable.
            key.extend_from_slice(nfd(value).as_bytes());
        }
        Some(key)
    }

    fn is_ignorable_unit(&self, unit: char) -> bool {
        // `alternate="blanked"` gives the variable characters no weight at any level, so a
        // substring match must step over them on both sides. `alternate="shifted"` only defers
        // them to the quaternary level, which a match at tertiary strength or below likewise
        // never looks at.
        match self.alternate {
            Alternate::NonIgnorable => false,
            Alternate::Blanked => self.is_variable(unit),
            Alternate::Shifted => self.strength < Strength::Quaternary && self.is_variable(unit),
        }
    }

    fn collation_units(&self, value: &str) -> Option<Vec<char>> {
        // One unit per *input* character, so that a match reports offsets in the original
        // string, folded to exactly the levels this collation compares at:
        //
        // * primary — the base of the character's own decomposition, case-folded, so both the
        //   accent and the case are invisible;
        // * secondary — the character itself, case-folded: `dâta` no longer matches `data`,
        //   because the accent is a *secondary* difference and this collation sees that level;
        // * tertiary and above (or any strength with `caseLevel=yes`) — the character as
        //   written, since case is significant too.
        //
        // A character the root table expands, or a tailoring re-letters, keeps its own unit: see
        // the note on expansions in the module documentation.
        Some(
            value
                .chars()
                .map(|character| {
                    let unit = if self.strength == Strength::Primary {
                        let decomposed = nfd(&character.to_string());
                        decomposed
                            .chars()
                            .find(|c| !weights::is_combining(*c))
                            .unwrap_or(character)
                    } else {
                        character
                    };
                    if self.strength <= Strength::Secondary && !self.case_level {
                        simple_fold(unit)
                    } else {
                        unit
                    }
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests;
