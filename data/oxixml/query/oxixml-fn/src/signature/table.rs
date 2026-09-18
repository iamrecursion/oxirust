//! The declared parameter types of the built-in functions, as F&O 3.1 spells them.
//!
//! The table covers the functions this crate's [`crate::registry`] serves, and the ones an
//! *evaluator* has to implement itself — the external-resource functions, which need an I/O
//! policy this crate deliberately has none of. A name and arity that is not here is left
//! unchecked, which is what makes the table safe to extend one function at a time.
//!
//! Every entry is the signature printed in the specification's function summary, with
//! `xs:numeric` kept as the union type F&O §1.6.2 defines rather than widened to `xs:double`.

use super::{Occurrence, ParamItemType, ParamType};
use crate::arrays::ARRAY_NAMESPACE;
use crate::maps::MAP_NAMESPACE;
use crate::registry::FN_NAMESPACE;
use oxixml_infoset::AtomicType;

const fn p(item_type: ParamItemType, occurrence: Occurrence) -> ParamType {
    ParamType::new(item_type, occurrence)
}

const fn atomic(atomic_type: AtomicType, occurrence: Occurrence) -> ParamType {
    p(ParamItemType::Atomic(atomic_type), occurrence)
}

const ITEM_Q: ParamType = p(ParamItemType::Item, Occurrence::ZeroOrOne);
const ITEM_S: ParamType = p(ParamItemType::Item, Occurrence::ZeroOrMore);
const NODE: ParamType = p(ParamItemType::Node, Occurrence::One);
const NODE_Q: ParamType = p(ParamItemType::Node, Occurrence::ZeroOrOne);
const NODE_S: ParamType = p(ParamItemType::Node, Occurrence::ZeroOrMore);
const MAP: ParamType = p(ParamItemType::Map, Occurrence::One);
const MAP_S: ParamType = p(ParamItemType::Map, Occurrence::ZeroOrMore);
const ARRAY: ParamType = p(ParamItemType::Array, Occurrence::One);
const ARRAY_S: ParamType = p(ParamItemType::Array, Occurrence::ZeroOrMore);
const NUMERIC_Q: ParamType = p(ParamItemType::Numeric, Occurrence::ZeroOrOne);

const STRING: ParamType = atomic(AtomicType::String, Occurrence::One);
const STRING_Q: ParamType = atomic(AtomicType::String, Occurrence::ZeroOrOne);
const STRING_S: ParamType = atomic(AtomicType::String, Occurrence::ZeroOrMore);
const INTEGER: ParamType = atomic(AtomicType::Integer, Occurrence::One);
const INTEGER_Q: ParamType = atomic(AtomicType::Integer, Occurrence::ZeroOrOne);
const INTEGER_S: ParamType = atomic(AtomicType::Integer, Occurrence::ZeroOrMore);
const DOUBLE: ParamType = atomic(AtomicType::Double, Occurrence::One);
const ANY: ParamType = atomic(AtomicType::AnyAtomicType, Occurrence::One);
const ANY_Q: ParamType = atomic(AtomicType::AnyAtomicType, Occurrence::ZeroOrOne);
const ANY_S: ParamType = atomic(AtomicType::AnyAtomicType, Occurrence::ZeroOrMore);
const QNAME: ParamType = atomic(AtomicType::QName, Occurrence::One);
const QNAME_Q: ParamType = atomic(AtomicType::QName, Occurrence::ZeroOrOne);
const BOOLEAN: ParamType = atomic(AtomicType::Boolean, Occurrence::One);
const BOOLEAN_Q: ParamType = atomic(AtomicType::Boolean, Occurrence::ZeroOrOne);
const ANYURI: ParamType = atomic(AtomicType::AnyUri, Occurrence::One);
const ANYURI_Q: ParamType = atomic(AtomicType::AnyUri, Occurrence::ZeroOrOne);
const NCNAME_Q: ParamType = atomic(AtomicType::Ncname, Occurrence::ZeroOrOne);
const DURATION_Q: ParamType = atomic(AtomicType::Duration, Occurrence::ZeroOrOne);
const DATETIME_Q: ParamType = atomic(AtomicType::DateTime, Occurrence::ZeroOrOne);
const DATE_Q: ParamType = atomic(AtomicType::Date, Occurrence::ZeroOrOne);
const TIME_Q: ParamType = atomic(AtomicType::Time, Occurrence::ZeroOrOne);
const DAYTIME_Q: ParamType = atomic(AtomicType::DayTimeDuration, Occurrence::ZeroOrOne);

/// `fn:concat` is variadic; every one of its arguments is `xs:anyAtomicType?`.
const CONCAT: &[ParamType] = &[
    ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q,
    ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q,
    ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q, ANY_Q,
];

/// The declared parameter types of `namespace:local#arity`, when this table knows them.
///
/// # Examples
///
/// ```
/// use oxixml_fn::signature::signature;
///
/// let ns = "http://www.w3.org/2005/xpath-functions";
/// assert_eq!(signature(ns, "substring", 3).map(<[_]>::len), Some(3));
/// assert!(signature(ns, "no-such-function", 1).is_none());
/// ```
#[must_use]
pub fn signature(namespace: &str, local: &str, arity: usize) -> Option<&'static [ParamType]> {
    match namespace {
        FN_NAMESPACE => fn_signature(local, arity),
        MAP_NAMESPACE => map_signature(local, arity),
        ARRAY_NAMESPACE => array_signature(local, arity),
        _ => None,
    }
}

/// The declared *result* type of `namespace:local#arity`, when this table knows it.
///
/// This is deliberately a separate, smaller table from [`signature`]. The parameter types are
/// needed for the function conversion rules, which every call goes through; a result type is
/// needed only to judge a function *item* against a `function(…) as R` item test, which is rarer
/// and which is answered on arity alone when the result type is unknown. Answering [`None`] is
/// therefore always safe, and the table can grow one function at a time.
///
/// # Examples
///
/// ```
/// use oxixml_fn::signature::result_type;
///
/// let ns = "http://www.w3.org/2005/xpath-functions";
/// assert!(result_type(ns, "name", 1).is_some());
/// assert!(result_type(ns, "no-such-function", 1).is_none());
/// ```
#[must_use]
pub fn result_type(namespace: &str, local: &str, arity: usize) -> Option<ParamType> {
    if namespace != FN_NAMESPACE {
        return None;
    }
    if local == "concat" {
        return (arity >= 2).then_some(STRING);
    }
    Some(match (local, arity) {
        // §5 accessors.
        ("node-name", 1) => QNAME_Q,
        ("nilled", 1) => BOOLEAN_Q,
        ("base-uri" | "document-uri", 1) => ANYURI_Q,
        ("string", 1) => STRING,
        ("data", 1) => ANY_S,

        // §4 numeric. `xs:numeric` is the union F&O §1.6.2 defines, and these functions return
        // the very member type they were given, so the result is the union too.
        ("abs" | "ceiling" | "floor" | "round" | "round-half-to-even", 1)
        | ("round" | "round-half-to-even", 2) => NUMERIC_Q,
        ("number", 1) => DOUBLE,
        ("format-integer", 2 | 3) | ("format-number", 2 | 3) => STRING,

        // §5, §6 strings and URIs.
        ("codepoints-to-string", 1) => STRING,
        ("string-to-codepoints", 1) => INTEGER_S,
        ("compare", 2 | 3) => INTEGER_Q,
        ("codepoint-equal", 2) => BOOLEAN_Q,
        ("string-join", 1 | 2) => STRING,
        ("substring", 2 | 3) => STRING,
        ("string-length", 1) => INTEGER,
        (
            "normalize-space" | "normalize-unicode" | "upper-case" | "lower-case"
            | "encode-for-uri" | "iri-to-uri" | "escape-html-uri",
            1,
        ) => STRING,
        ("normalize-unicode", 2) => STRING,
        ("translate", 3) => STRING,
        ("contains" | "starts-with" | "ends-with", 2 | 3) => BOOLEAN,
        ("substring-before" | "substring-after", 2 | 3) => STRING,
        ("matches", 2 | 3) => BOOLEAN,
        ("replace", 3 | 4) => STRING,
        ("tokenize", 1..=3) => STRING_S,

        // §7 booleans.
        ("true" | "false", 0) => BOOLEAN,
        ("boolean" | "not", 1) => BOOLEAN,

        // §11 QNames.
        ("resolve-QName", 2) => QNAME_Q,
        ("QName", 2) => QNAME,
        ("prefix-from-QName", 1) => NCNAME_Q,
        ("local-name-from-QName", 1) => NCNAME_Q,
        ("namespace-uri-from-QName", 1) => ANYURI_Q,
        ("namespace-uri-for-prefix", 2) => ANYURI_Q,
        ("in-scope-prefixes", 1) => STRING_S,

        // §14 nodes.
        ("name" | "local-name" | "generate-id", 1) => STRING,
        ("namespace-uri", 1) => ANYURI,
        ("root", 1) => NODE_Q,
        ("has-children", 1) => BOOLEAN,
        ("path", 1) => STRING_Q,
        ("lang", 2) => BOOLEAN,
        // `fn:id` and `fn:element-with-id` are declared `element()*`, which this table has no
        // item type for; declaring the wider `node()*` would be a *wrong* signature rather than
        // an absent one, so they are left out.
        ("innermost" | "outermost", 1) => NODE_S,

        // §14 sequences.
        ("empty" | "exists", 1) => BOOLEAN,
        ("count", 1) => INTEGER,
        ("head", 1) => ITEM_Q,
        ("reverse" | "unordered" | "tail", 1) => ITEM_S,
        _ => return None,
    })
}

#[allow(clippy::too_many_lines)]
fn fn_signature(local: &str, arity: usize) -> Option<&'static [ParamType]> {
    if local == "concat" {
        return (arity >= 2).then(|| &CONCAT[..arity.min(CONCAT.len())]);
    }
    Some(match (local, arity) {
        // §5 accessors.
        ("node-name" | "nilled" | "base-uri" | "document-uri", 1) => &[NODE_Q],
        ("string", 1) => &[ITEM_Q],
        ("data", 1) => &[ITEM_S],
        ("trace", 2) => &[ITEM_S, STRING],
        ("error", 1) => &[QNAME_Q],
        ("error", 2) => &[QNAME_Q, STRING],
        ("error", 3) => &[QNAME_Q, STRING, ITEM_S],

        // §4 numeric.
        ("abs" | "ceiling" | "floor" | "round" | "round-half-to-even", 1) => &[NUMERIC_Q],
        ("round" | "round-half-to-even", 2) => &[NUMERIC_Q, INTEGER],
        ("number", 1) => &[ANY_Q],
        ("format-integer", 2) => &[INTEGER_Q, STRING],
        ("format-integer", 3) => &[INTEGER_Q, STRING, STRING_Q],
        ("format-number", 2) => &[NUMERIC_Q, STRING],
        ("format-number", 3) => &[NUMERIC_Q, STRING, STRING_Q],

        // §5, §6 strings and URIs.
        ("codepoints-to-string", 1) => &[INTEGER_S],
        ("string-to-codepoints", 1) => &[STRING_Q],
        ("compare" | "codepoint-equal", 2) => &[STRING_Q, STRING_Q],
        ("compare", 3) => &[STRING_Q, STRING_Q, STRING],
        // F&O 3.1 widened *both* forms to `xs:anyAtomicType*` (W3C bug 29184): `string-join(1 to
        // 5, '')` is `12345` from 3.1 onwards (`fn-string-join-28`, `-31`) where 2.0 and 3.0
        // made it `XPTY0004` (`fn-string-join-27`, which is pinned to those versions).
        ("string-join", 1) => &[ANY_S],
        ("string-join", 2) => &[ANY_S, STRING],
        ("substring", 2) => &[STRING_Q, DOUBLE],
        ("substring", 3) => &[STRING_Q, DOUBLE, DOUBLE],
        (
            "string-length" | "normalize-space" | "normalize-unicode" | "upper-case" | "lower-case"
            | "encode-for-uri" | "iri-to-uri" | "escape-html-uri" | "tokenize",
            1,
        ) => &[STRING_Q],
        ("normalize-unicode", 2) => &[STRING_Q, STRING],
        ("translate", 3) => &[STRING_Q, STRING, STRING],
        ("contains" | "starts-with" | "ends-with" | "substring-before" | "substring-after", 2) => {
            &[STRING_Q, STRING_Q]
        }
        ("contains" | "starts-with" | "ends-with" | "substring-before" | "substring-after", 3) => {
            &[STRING_Q, STRING_Q, STRING]
        }
        ("matches" | "tokenize" | "analyze-string", 2) => &[STRING_Q, STRING],
        ("matches" | "tokenize" | "analyze-string", 3) => &[STRING_Q, STRING, STRING],
        ("replace", 3) => &[STRING_Q, STRING, STRING],
        ("replace", 4) => &[STRING_Q, STRING, STRING, STRING],
        ("resolve-uri", 1) => &[STRING_Q],
        ("resolve-uri", 2) => &[STRING_Q, STRING],
        ("contains-token", 2) => &[STRING_S, STRING],
        ("contains-token", 3) => &[STRING_S, STRING, STRING],
        ("collation-key", 1) => &[STRING],
        ("collation-key", 2) => &[STRING, STRING],

        // §7 boolean.
        ("boolean" | "not", 1) => &[ITEM_S],

        // §9 durations, dates and times.
        (
            "years-from-duration"
            | "months-from-duration"
            | "days-from-duration"
            | "hours-from-duration"
            | "minutes-from-duration"
            | "seconds-from-duration",
            1,
        ) => &[DURATION_Q],
        (
            "year-from-dateTime"
            | "month-from-dateTime"
            | "day-from-dateTime"
            | "hours-from-dateTime"
            | "minutes-from-dateTime"
            | "seconds-from-dateTime"
            | "timezone-from-dateTime"
            | "adjust-dateTime-to-timezone",
            1,
        ) => &[DATETIME_Q],
        (
            "year-from-date"
            | "month-from-date"
            | "day-from-date"
            | "timezone-from-date"
            | "adjust-date-to-timezone",
            1,
        ) => &[DATE_Q],
        (
            "hours-from-time"
            | "minutes-from-time"
            | "seconds-from-time"
            | "timezone-from-time"
            | "adjust-time-to-timezone",
            1,
        ) => &[TIME_Q],
        ("adjust-dateTime-to-timezone", 2) => &[DATETIME_Q, DAYTIME_Q],
        ("adjust-date-to-timezone", 2) => &[DATE_Q, DAYTIME_Q],
        ("adjust-time-to-timezone", 2) => &[TIME_Q, DAYTIME_Q],
        ("dateTime", 2) => &[DATE_Q, TIME_Q],
        ("format-dateTime", 2) => &[DATETIME_Q, STRING],
        ("format-dateTime", 5) => &[DATETIME_Q, STRING, STRING_Q, STRING_Q, STRING_Q],
        ("format-date", 2) => &[DATE_Q, STRING],
        ("format-date", 5) => &[DATE_Q, STRING, STRING_Q, STRING_Q, STRING_Q],
        ("format-time", 2) => &[TIME_Q, STRING],
        ("format-time", 5) => &[TIME_Q, STRING, STRING_Q, STRING_Q, STRING_Q],
        ("parse-ietf-date", 1) => &[STRING_Q],

        // §11 QNames.
        ("resolve-QName", 2) => &[STRING_Q, NODE],
        ("QName", 2) => &[STRING_Q, STRING],
        ("prefix-from-QName" | "local-name-from-QName" | "namespace-uri-from-QName", 1) => {
            &[QNAME_Q]
        }
        ("namespace-uri-for-prefix", 2) => &[STRING_Q, NODE],
        ("in-scope-prefixes", 1) => &[NODE],

        // §14 nodes.
        (
            "name" | "local-name" | "namespace-uri" | "root" | "has-children" | "generate-id"
            | "path",
            1,
        ) => &[NODE_Q],
        ("lang", 2) => &[STRING_Q, NODE],
        ("id" | "idref" | "element-with-id", 2) => &[STRING_S, NODE],
        ("innermost" | "outermost", 1) => &[NODE_S],

        // §14 sequences.
        (
            "empty" | "exists" | "head" | "tail" | "reverse" | "unordered" | "count"
            | "zero-or-one" | "one-or-more" | "exactly-one",
            1,
        ) => &[ITEM_S],
        ("insert-before", 3) => &[ITEM_S, INTEGER, ITEM_S],
        ("remove", 2) => &[ITEM_S, INTEGER],
        ("subsequence", 2) => &[ITEM_S, DOUBLE],
        ("subsequence", 3) => &[ITEM_S, DOUBLE, DOUBLE],
        ("index-of", 2) => &[ANY_S, ANY],
        ("index-of", 3) => &[ANY_S, ANY, STRING],
        ("distinct-values" | "max" | "min", 1) => &[ANY_S],
        ("distinct-values" | "max" | "min", 2) => &[ANY_S, STRING],
        ("deep-equal", 2) => &[ITEM_S, ITEM_S],
        ("deep-equal", 3) => &[ITEM_S, ITEM_S, STRING],
        ("avg", 1) => &[ANY_S],
        ("sum", 1) => &[ANY_S],
        ("sum", 2) => &[ANY_S, ANY_Q],

        // §13, §14.7 parsing and serializing.
        ("parse-xml" | "parse-xml-fragment", 1) => &[STRING_Q],
        ("serialize", 1) => &[ITEM_S],
        ("serialize", 2) => &[ITEM_S, ITEM_Q],
        ("parse-json" | "json-to-xml", 1) => &[STRING_Q],
        ("parse-json" | "json-to-xml", 2) => &[STRING_Q, MAP],
        ("xml-to-json", 1) => &[NODE_Q],
        ("xml-to-json", 2) => &[NODE_Q, MAP],

        // §14.5 external resources. This crate does not implement them — they need an I/O
        // policy — but their argument types are part of the same specification, and the
        // evaluator that does implement them consults this table.
        ("doc" | "doc-available" | "collection" | "uri-collection" | "json-doc", 1) => &[STRING_Q],
        ("json-doc", 2) => &[STRING_Q, MAP],
        ("unparsed-text" | "unparsed-text-lines" | "unparsed-text-available", 1) => &[STRING_Q],
        ("unparsed-text" | "unparsed-text-lines" | "unparsed-text-available", 2) => {
            &[STRING_Q, STRING]
        }
        ("environment-variable", 1) => &[STRING],
        ("function-lookup", 2) => &[QNAME, INTEGER],
        ("random-number-generator", 1) => &[ANY_Q],
        _ => return None,
    })
}

fn map_signature(local: &str, arity: usize) -> Option<&'static [ParamType]> {
    Some(match (local, arity) {
        ("size" | "keys", 1) => &[MAP],
        ("contains" | "get", 2) => &[MAP, ANY],
        ("put", 3) => &[MAP, ANY, ITEM_S],
        ("remove", 2) => &[MAP, ANY_S],
        ("entry", 2) => &[ANY, ITEM_S],
        ("merge", 1) => &[MAP_S],
        ("merge", 2) => &[MAP_S, MAP],
        ("find", 2) => &[ITEM_S, ANY],
        _ => return None,
    })
}

fn array_signature(local: &str, arity: usize) -> Option<&'static [ParamType]> {
    Some(match (local, arity) {
        ("size" | "head" | "tail" | "reverse", 1) => &[ARRAY],
        ("get", 2) => &[ARRAY, INTEGER],
        ("put", 3) => &[ARRAY, INTEGER, ITEM_S],
        ("append", 2) => &[ARRAY, ITEM_S],
        ("subarray", 2) => &[ARRAY, INTEGER],
        ("subarray", 3) => &[ARRAY, INTEGER, INTEGER],
        ("remove", 2) => &[ARRAY, INTEGER_S],
        ("insert-before", 3) => &[ARRAY, INTEGER, ITEM_S],
        ("join", 1) => &[ARRAY_S],
        ("flatten", 1) => &[ITEM_S],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_answers_by_name_and_arity() {
        assert_eq!(
            signature(FN_NAMESPACE, "substring", 2).map(<[_]>::len),
            Some(2)
        );
        assert_eq!(
            signature(FN_NAMESPACE, "substring", 3).map(<[_]>::len),
            Some(3)
        );
        assert!(signature(FN_NAMESPACE, "substring", 4).is_none());
        assert_eq!(
            signature(FN_NAMESPACE, "concat", 7).map(<[_]>::len),
            Some(7)
        );
        assert!(signature("urn:example", "substring", 2).is_none());
    }

    #[test]
    fn every_entry_declares_one_type_per_argument() {
        for arity in 0..=5 {
            for local in [
                "substring",
                "replace",
                "round",
                "index-of",
                "format-date",
                "map:merge",
            ] {
                if let Some(parameters) = signature(FN_NAMESPACE, local, arity) {
                    assert_eq!(parameters.len(), arity, "{local}#{arity}");
                }
            }
        }
    }
}
