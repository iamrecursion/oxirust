//! `xsl:sort`: multi-key sorting of a sequence, and of the groups `xsl:for-each-group` forms.
//!
//! [XSLT 3.0 §13.1](https://www.w3.org/TR/xslt-30/#sorting) makes each `xsl:sort` a *sort key
//! component*: an expression evaluated once per item with that item in focus, an order, a data
//! type and a collation. The sort is stable unless `stable="no"` is written, which is what
//! Rust's `sort_by` gives for free.

use super::{Context, Transformer};
use crate::compile::SortKey;
use crate::error::XsltResult;
use oxixml_infoset::{AtomicValue, Item, Sequence, compare_atomic_in_context};
use oxixml_xpath::XpathValue;
use oxixml_xsd::TimezoneOffset;
use std::cmp::Ordering;

/// One item's computed sort keys.
struct Keyed {
    item: Item,
    keys: Vec<KeyValue>,
}

/// A single sort key value.
enum KeyValue {
    /// An empty sequence — the "empty least" case.
    Empty,
    /// A number: either `data-type="number"` was written, or the atomized key is numeric.
    Number(f64),
    /// A string, compared under the key's collation.
    Text(String),
    /// A boolean, `false` before `true`.
    Boolean(bool),
    /// A value of some other atomic type — a date, a duration — held alongside the name of its
    /// type, because only values of the same type are comparable. Kept as the typed value
    /// itself, not its lexical text, so that ordering follows F&O's `op:*-less-than` (which for
    /// date/time types reads a missing timezone in the dynamic context's implicit timezone)
    /// rather than a plain string comparison — dates in particular do not always sort the same
    /// way lexically as they do by value (`"2000-01-01T00:00:00+01:00"` spells greater than
    /// `"1999-12-31T23:30:00Z"` but denotes an earlier instant).
    Other(AtomicValue, String),
}

impl KeyValue {
    /// The comparison class a value belongs to.
    ///
    /// [XSLT 3.0 §13.1](https://www.w3.org/TR/xslt-30/#sorting) compares sort key values with
    /// the XPath `lt` operator, and makes it `XTDE1030` if that operator raises an error for
    /// any pair — which is exactly what happens when one key is a number and another a string.
    fn class(&self) -> Option<&str> {
        match self {
            Self::Empty => None,
            Self::Number(_) => Some("numeric"),
            Self::Text(_) => Some("string"),
            Self::Boolean(_) => Some("boolean"),
            Self::Other(_, name) => Some(name),
        }
    }
}

/// How one key component is compared.
struct KeySpec {
    descending: bool,
    numeric: bool,
    /// `data-type="text"`, which makes the key a string however the atomized value is typed.
    textual: bool,
    /// `Some(lower_first)` when the `case-order` attribute was written, `None` when it was not.
    case_order: Option<bool>,
    stable: bool,
    collation: std::sync::Arc<dyn oxixml_fn::Collation>,
    /// The dynamic context's [implicit
    /// timezone](https://www.w3.org/TR/xpath-functions-31/#dt-implicit-timezone), which a typed
    /// date/time key with no timezone of its own is read in before comparing (F&O 3.1 §10.4.4).
    implicit_timezone: TimezoneOffset,
}

impl Transformer {
    /// Sorts `items` by `keys`.
    ///
    /// # Errors
    ///
    /// Any dynamic error a key expression raises.
    pub(crate) fn sort_items(
        &mut self,
        items: Vec<Item>,
        keys: &[SortKey],
        context: &Context,
    ) -> XsltResult<Vec<Item>> {
        if keys.is_empty() {
            return Ok(items);
        }
        // The key specifications are evaluated even when there is nothing to sort: an
        // unrecognized `collation` is `XTDE1035` and an ill-formed `lang` is `XTDE0030`
        // whatever the length of the sequence.
        let specs = self.key_specs(keys, context)?;
        if items.len() < 2 {
            return Ok(items);
        }
        let size = items.len();
        let mut keyed = Vec::with_capacity(size);
        for (offset, item) in items.into_iter().enumerate() {
            let scoped = context.with_focus(item.clone(), offset + 1, size);
            let mut values = Vec::with_capacity(keys.len());
            for (key, spec) in keys.iter().zip(specs.iter()) {
                values.push(self.key_value(key, spec, &scoped)?);
            }
            keyed.push(Keyed { item, keys: values });
        }
        for component in 0..keys.len() {
            let values: Vec<&KeyValue> = keyed
                .iter()
                .filter_map(|entry| entry.keys.get(component))
                .collect();
            check_comparable_refs(&values)?;
        }
        if specs.iter().all(|spec| spec.stable) {
            keyed.sort_by(|left, right| compare_keys(&left.keys, &right.keys, &specs));
        } else {
            keyed.sort_unstable_by(|left, right| compare_keys(&left.keys, &right.keys, &specs));
        }
        Ok(keyed.into_iter().map(|entry| entry.item).collect())
    }

    /// Sorts the groups `xsl:for-each-group` formed, with each group's first item in focus.
    ///
    /// # Errors
    ///
    /// Any dynamic error a key expression raises.
    pub(crate) fn sort_groups(
        &mut self,
        groups: Vec<(Sequence, Vec<Item>)>,
        keys: &[SortKey],
        context: &Context,
    ) -> XsltResult<Vec<(Sequence, Vec<Item>)>> {
        if keys.is_empty() {
            return Ok(groups);
        }
        let specs = self.key_specs(keys, context)?;
        if groups.len() < 2 {
            return Ok(groups);
        }
        let size = groups.len();
        let mut keyed = Vec::with_capacity(size);
        for (offset, (key, members)) in groups.into_iter().enumerate() {
            let mut values = Vec::with_capacity(keys.len());
            let focus = members.first().cloned();
            self.runtime
                .groups
                .borrow_mut()
                .push((members.clone(), Some(key.clone())));
            let scoped = match focus {
                Some(item) => context.with_focus(item, offset + 1, size),
                None => context.clone(),
            };
            let mut failure = None;
            for (sort_key, spec) in keys.iter().zip(specs.iter()) {
                match self.key_value(sort_key, spec, &scoped) {
                    Ok(value) => values.push(value),
                    Err(error) => {
                        failure = Some(error);
                        break;
                    }
                }
            }
            self.runtime.groups.borrow_mut().pop();
            if let Some(error) = failure {
                return Err(error);
            }
            keyed.push((values, key, members));
        }
        keyed.sort_by(|left, right| compare_keys(&left.0, &right.0, &specs));
        Ok(keyed
            .into_iter()
            .map(|(_, key, members)| (key, members))
            .collect())
    }

    fn key_specs(&mut self, keys: &[SortKey], context: &Context) -> XsltResult<Vec<KeySpec>> {
        let mut specs = Vec::with_capacity(keys.len());
        for key in keys {
            let order = self.evaluate_avt(&key.order, context)?;
            let data_type = match &key.data_type {
                Some(avt) => Some(self.evaluate_avt(avt, context)?),
                None => None,
            };
            let data_type = data_type.as_deref().map(str::trim);
            let case_order = match &key.case_order {
                Some(avt) => Some(self.evaluate_avt(avt, context)?.trim() == "lower-first"),
                None => None,
            };
            let mut explicit = false;
            let mut case_order_in_collation = false;
            let collation = match (&key.collation, &key.language) {
                (Some(avt), _) => {
                    explicit = true;
                    Some(self.evaluate_avt(avt, context)?)
                }
                // `lang` selects a collation only when one has been registered under the
                // language's URI; otherwise the default collation stands.
                (None, Some(avt)) => {
                    let language = self.evaluate_avt(avt, context)?;
                    if !crate::names::is_language(language.trim()) {
                        return Err(crate::error::XsltError::dynamic(
                            "XTDE0030",
                            format!("`{language}` is not a value for the `lang` attribute"),
                        ));
                    }
                    (!language.is_empty()).then(|| {
                        let mut uri =
                            format!("http://www.w3.org/2013/collation/UCA?lang={language}");
                        // The language-sensitive collation orders case itself, and `caseFirst`
                        // is the parameter that says which way round. Leaving `case-order` as a
                        // tie-break instead would make it dead: a tertiary-strength UCA
                        // collation never calls `b` and `B` equal, so the tie never arises
                        // (`sort-018`).
                        if let Some(lower_first) = case_order {
                            case_order_in_collation = true;
                            uri.push_str(if lower_first {
                                ";caseFirst=lower"
                            } else {
                                ";caseFirst=upper"
                            });
                        }
                        uri
                    })
                }
                // XSLT 3.0 §13.1: with neither attribute written the default collation applies,
                // and §5.4.4 makes that the one in force where the `xsl:sort` stands.
                (None, None) => key.default_collation.clone(),
            };
            // XSLT 3.0 §13.1: a `collation` this processor does not recognize is `XTDE1035`.
            // A collation derived from `lang` is a hint rather than a request, so an
            // unrecognized language falls back to the default collation instead.
            let collation = match self.runtime.fn_context.collation(collation.as_deref()) {
                Ok(found) => found,
                Err(_) if !explicit => {
                    // The language-derived collation was refused, so the case ordering that was
                    // folded into it is not in force after all and the tie-break has to stand.
                    case_order_in_collation = false;
                    self.runtime.fn_context.default_collation().clone()
                }
                Err(_) => {
                    return Err(crate::error::XsltError::dynamic(
                        "XTDE1035",
                        format!(
                            "`{}` names no collation this processor recognizes",
                            collation.unwrap_or_default()
                        ),
                    ));
                }
            };
            specs.push(KeySpec {
                descending: order.trim() == "descending",
                numeric: data_type == Some("number"),
                // XSLT 3.0 §13.1.2 keeps `data-type` "for backwards compatibility with XSLT
                // 1.0", where every sort key was a string: written out, `text` means the key is
                // compared as a string whatever its type, so a numeric `select` sorts
                // `12, 15, 18, 20, 21, 24, 8` (`sort-026`). Leaving the attribute out is a
                // different instruction, and one only XSLT 1.0 spells `text`.
                textual: data_type == Some("text"),
                // XSLT 3.0 §13.1.3: `case-order` is ignored when the sort names a `collation`,
                // whose ordering is then the whole answer — `sort-067` writes both and expects
                // plain code-point order — and it is equally spent once it has been folded into
                // a language-derived collation as `caseFirst`.
                case_order: (!explicit && !case_order_in_collation)
                    .then_some(case_order)
                    .flatten(),
                stable: key.stable,
                collation,
                implicit_timezone: self.runtime.fn_context.implicit_timezone(),
            });
        }
        Ok(specs)
    }

    fn key_value(
        &mut self,
        key: &SortKey,
        spec: &KeySpec,
        context: &Context,
    ) -> XsltResult<KeyValue> {
        // XSLT 3.0 §13.1: the sort key value is the *atomized* result of the `select`
        // expression or, where the element carries one instead, of its sequence constructor.
        // The constructor's result is a value like any other, not a string: `sort-051` builds
        // its key with `<xsl:sequence select="round(.)"/>` and asks for the rounded numbers in
        // numeric order, which the string form `803.23 < 99` does not give.
        let value = match &key.select {
            Some(select) => Some(self.evaluate(select, context)?),
            None if key.content.is_empty() => None,
            None => Some(XpathValue::from(
                self.run_to_sequence(&key.content, None, context)?.1,
            )),
        };
        let mut typed: Option<oxixml_infoset::AtomicValue> = None;
        let text = match value {
            Some(value) => {
                if value.is_empty() {
                    return Ok(KeyValue::Empty);
                }
                // XSLT 3.0 §13.1: outside XSLT 1.0 behaviour a sort key value of more than one
                // item, after atomization, is the type error `XTTE1020`.
                let atomized = value.atomize()?;
                if atomized.len() > 1 && self.ir().version != crate::compile::XsltVersion::V10 {
                    return Err(crate::error::XsltError::dynamic(
                        "XTTE1020",
                        format!(
                            "a sort key produced {} items, and only one is allowed",
                            atomized.len()
                        ),
                    ));
                }
                typed = atomized.first().cloned().filter(|_| atomized.len() == 1);
                let mut parts = Vec::with_capacity(value.len());
                for item in value.iter() {
                    parts.push(item.string_value()?);
                }
                parts.join(" ")
            }
            None => context
                .item
                .as_ref()
                .map(oxixml_infoset::Item::string_value)
                .transpose()?
                .unwrap_or_default(),
        };
        if spec.numeric {
            return Ok(KeyValue::Number(parse_number(&text)));
        }
        if spec.textual {
            return Ok(KeyValue::Text(text));
        }
        Ok(match typed {
            // XSLT 3.0 §13.1: with no `data-type`, the atomized sort key keeps its own type and
            // is compared the way the XPath `lt` operator would compare it. `xs:untypedAtomic`
            // — the type of every unvalidated node — is compared as a string.
            Some(value) if value.is_numeric() => {
                KeyValue::Number(value.as_f64().unwrap_or(f64::NAN))
            }
            Some(value) if value.atomic_type() == oxixml_infoset::AtomicType::Boolean => {
                KeyValue::Boolean(value.as_bool().unwrap_or(false))
            }
            Some(value) if !is_string_like(value.atomic_type()) => {
                let type_name = format!("{:?}", value.atomic_type());
                KeyValue::Other(value, type_name)
            }
            _ => KeyValue::Text(text),
        })
    }
}

/// Rejects a sort key component whose values are not all mutually comparable.
///
/// # Errors
///
/// `XTDE1030`, which [XSLT 3.0 §13.1](https://www.w3.org/TR/xslt-30/#sorting) raises when the
/// `lt` operator would fail on a pair of the component's values.
fn check_comparable_refs(values: &[&KeyValue]) -> XsltResult<()> {
    let mut seen: Option<&str> = None;
    for value in values {
        // Sharing a type is not enough: F&O 3.1 defines `lt` only for the types it orders, and
        // for the rest — `xs:duration`, whose months and seconds are incommensurable
        // (`date-053` sorts `P1Y` against `P365D`), the Gregorian types, the binary types and
        // `xs:QName` — comparing two values of the very same type is still an error.
        if let KeyValue::Other(atomic, name) = value
            && !is_ordered(atomic.atomic_type())
        {
            return Err(crate::error::XsltError::dynamic(
                "XTDE1030",
                format!("sort key values of type {name} cannot be compared with `lt`"),
            ));
        }
        let Some(class) = value.class() else { continue };
        match seen {
            Some(found) if found != class => {
                return Err(crate::error::XsltError::dynamic(
                    "XTDE1030",
                    format!(
                        "a sort key mixes {found} and {class} values, which cannot be compared"
                    ),
                ));
            }
            _ => seen = Some(class),
        }
    }
    Ok(())
}

/// Whether F&O 3.1 gives `type` a `lt` operator.
///
/// [F&O 3.1 §8.3](https://www.w3.org/TR/xpath-functions-31/#comparison-ops-on-durations) orders
/// `xs:yearMonthDuration` and `xs:dayTimeDuration` but not their common ancestor `xs:duration`,
/// and [§9.4](https://www.w3.org/TR/xpath-functions-31/#comp.datetime) leaves the Gregorian
/// types, the binary types and `xs:QName` with equality alone. Sorting on such a key is
/// `XTDE1030` however uniform the values are.
fn is_ordered(atomic_type: oxixml_infoset::AtomicType) -> bool {
    use oxixml_infoset::AtomicType;
    !matches!(
        atomic_type,
        AtomicType::Duration
            | AtomicType::GYearMonth
            | AtomicType::GYear
            | AtomicType::GMonthDay
            | AtomicType::GMonth
            | AtomicType::GDay
            | AtomicType::HexBinary
            | AtomicType::Base64Binary
            | AtomicType::QName
            | AtomicType::Notation
    )
}

fn parse_number(text: &str) -> f64 {
    text.trim().parse::<f64>().unwrap_or(f64::NAN)
}

fn compare_keys(left: &[KeyValue], right: &[KeyValue], specs: &[KeySpec]) -> Ordering {
    for ((left, right), spec) in left.iter().zip(right.iter()).zip(specs.iter()) {
        let ordering = compare_one(left, right, spec);
        if ordering != Ordering::Equal {
            return if spec.descending {
                ordering.reverse()
            } else {
                ordering
            };
        }
    }
    Ordering::Equal
}

fn compare_one(left: &KeyValue, right: &KeyValue, spec: &KeySpec) -> Ordering {
    match (left, right) {
        (KeyValue::Empty, KeyValue::Empty) => Ordering::Equal,
        (KeyValue::Empty, _) => Ordering::Less,
        (_, KeyValue::Empty) => Ordering::Greater,
        (KeyValue::Number(left), KeyValue::Number(right)) => {
            match (left.is_nan(), right.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                (false, false) => left.partial_cmp(right).unwrap_or(Ordering::Equal),
            }
        }
        (KeyValue::Text(left), KeyValue::Text(right)) => {
            // XSLT 3.0 §13.1.3 defines `case-order` as a tie-break between strings the
            // collation calls equal — which assumes a language-sensitive collation. The
            // processor's default is the code point collation, which never calls `a` and `A`
            // equal, so the attribute would be inert. When it is written and the collation is
            // the code point one, the comparison is made case-blind first and `case-order`
            // then decides, which is what the attribute is for.
            if let Some(lower_first) = spec.case_order {
                if spec.collation.uri() == oxixml_fn::CODEPOINT_COLLATION {
                    let ordering = compare_case_blind(left, right);
                    if ordering != Ordering::Equal {
                        return ordering;
                    }
                    return case_order(left, right, lower_first);
                }
            }
            let ordering = spec.collation.compare(left, right);
            if ordering != Ordering::Equal {
                return ordering;
            }
            // Case order only decides between strings the collation calls equal, and only when
            // the stylesheet asked for it. With the attribute absent nothing may break the tie:
            // XSLT 3.0 §13.1 makes the sort stable by default, so two spellings of one word that
            // a case-blind collation calls equal keep the order the population had.
            match spec.case_order {
                Some(lower_first) => case_order(left, right, lower_first),
                None => Ordering::Equal,
            }
        }
        (KeyValue::Boolean(left), KeyValue::Boolean(right)) => left.cmp(right),
        // F&O 3.1 §10.4.4: a date/time value with no timezone of its own is read in the dynamic
        // context's implicit timezone before comparing, exactly as the `lt` operator reads it.
        // `check_comparable_refs` has already confirmed both sides share the same underlying
        // type, so `Ok(None)` (only possible for the genuinely partial order among
        // `xs:duration` values) and the unreachable `Err` fall back to `Equal`, matching how the
        // `Number` arm above treats its own incomparable case (`NaN`).
        (KeyValue::Other(left, _), KeyValue::Other(right, _)) => {
            compare_atomic_in_context(left, right, None, Some(spec.implicit_timezone))
                .ok()
                .flatten()
                .unwrap_or(Ordering::Equal)
        }
        // Mixed classes never reach here: `check_comparable` has already raised `XTDE1030`.
        (left, right) => {
            let left = parse_number(&left.text());
            let right = parse_number(&right.text());
            left.partial_cmp(&right).unwrap_or(Ordering::Equal)
        }
    }
}

impl KeyValue {
    fn text(&self) -> String {
        match self {
            Self::Empty => String::new(),
            Self::Number(value) => value.to_string(),
            Self::Text(value) => value.clone(),
            Self::Boolean(value) => value.to_string(),
            Self::Other(value, _) => value.string_value(),
        }
    }
}

/// Whether a type's values are compared as strings, which every string-derived type and
/// `xs:untypedAtomic` are.
fn is_string_like(atomic_type: oxixml_infoset::AtomicType) -> bool {
    use oxixml_infoset::AtomicType;
    !matches!(
        atomic_type,
        AtomicType::Date
            | AtomicType::DateTime
            | AtomicType::DateTimeStamp
            | AtomicType::Time
            | AtomicType::Duration
            | AtomicType::YearMonthDuration
            | AtomicType::DayTimeDuration
            | AtomicType::GYear
            | AtomicType::GYearMonth
            | AtomicType::GMonth
            | AtomicType::GMonthDay
            | AtomicType::GDay
            | AtomicType::HexBinary
            | AtomicType::Base64Binary
            | AtomicType::QName
            | AtomicType::Notation
    )
}

/// Orders two strings by code point after lower-casing, so that a case pair ties.
fn compare_case_blind(left: &str, right: &str) -> Ordering {
    left.chars()
        .flat_map(char::to_lowercase)
        .cmp(right.chars().flat_map(char::to_lowercase))
}

fn case_order(left: &str, right: &str, lower_first: bool) -> Ordering {
    for (left, right) in left.chars().zip(right.chars()) {
        if left == right {
            continue;
        }
        if left.to_lowercase().eq(right.to_lowercase()) {
            let left_is_lower = left.is_lowercase();
            return match (left_is_lower, lower_first) {
                (true, true) | (false, false) => Ordering::Less,
                _ => Ordering::Greater,
            };
        }
        return Ordering::Equal;
    }
    Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_that_do_not_parse_sort_first() {
        assert!(parse_number("nonsense").is_nan());
        assert_eq!(parse_number(" 4 "), 4.0);
    }

    #[test]
    fn case_order_only_breaks_ties() {
        assert_eq!(case_order("a", "A", false), Ordering::Greater);
        assert_eq!(case_order("a", "A", true), Ordering::Less);
        assert_eq!(case_order("ab", "ac", true), Ordering::Equal);
    }
}
