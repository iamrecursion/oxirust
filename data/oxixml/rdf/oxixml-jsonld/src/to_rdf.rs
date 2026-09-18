//! [Deserialize JSON-LD to RDF](https://www.w3.org/TR/json-ld-api/#deserialize-json-ld-to-rdf-algorithm):
//! turns an expanded document's node map into a set of RDF [`Quad`]s.

use crate::value::{JsonObject, JsonValue};
use oxixml_model::vocab::{rdf, xsd};
use oxixml_model::{BlankNode, GraphName, Literal, NamedNode, NamedOrBlankNode, Quad, Term};

/// How `@direction` is represented in the produced RDF.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum RdfDirection {
    /// `@direction` produces no RDF (the default).
    #[default]
    None,
    /// `@direction` is folded into a `https://www.w3.org/ns/i18n#` datatype.
    I18nDatatype,
    /// `@direction` produces a compound literal (a blank node with `rdf:value`, `rdf:language`,
    /// `rdf:direction`).
    CompoundLiteral,
}

/// Options controlling RDF serialization.
#[derive(Clone, Copy, Debug, Default)]
pub struct ToRdfOptions {
    /// Whether to emit generalized RDF (blank nodes as predicates).
    pub produce_generalized_rdf: bool,
    /// How to represent `@direction`.
    pub rdf_direction: RdfDirection,
}

/// Converts a node map into RDF quads.
pub struct RdfConverter {
    options: ToRdfOptions,
    quads: Vec<Quad>,
    list_counter: usize,
}

impl RdfConverter {
    /// Creates a converter with the given options.
    #[must_use]
    pub fn new(options: ToRdfOptions) -> Self {
        Self {
            options,
            quads: Vec::new(),
            list_counter: 0,
        }
    }

    /// Runs the deserialization algorithm over `node_map`, returning the quads.
    ///
    /// This mirrors the [Deserialize JSON-LD to RDF
    /// algorithm](https://www.w3.org/TR/json-ld-api/#deserialize-json-ld-to-rdf-algorithm)
    /// exactly: every value encountered while walking the node map contributes its own quad, even
    /// when two *different* source values denote the same RDF term (say a native `true` under one
    /// `@index` container key and another native `true` under a different key). No
    /// deduplication happens here — the W3C `toRdf` conformance suite depends on that: test
    /// `#te036` ("Expanding `@index`") puts a native `true` and a native `false` under two
    /// different `@index`-container keys, and its expected `e036-out.nq` contains each of the
    /// resulting quads *twice*. `toRdf` output is compared by the suite as a multiset of N-Quads
    /// lines, not as an RDF-graph isomorphism, so collapsing repeats here would silently drop
    /// conformant quads.
    #[must_use]
    pub fn convert(self, node_map: &JsonObject) -> Vec<Quad> {
        self.convert_all(node_map)
    }

    fn convert_all(mut self, node_map: &JsonObject) -> Vec<Quad> {
        for (graph_name, graph) in node_map.iter() {
            let Some(graph) = graph.as_object() else {
                continue;
            };
            let graph_term = if graph_name == "@default" {
                Some(GraphName::DefaultGraph)
            } else {
                self.to_named_or_blank(graph_name).map(Into::into)
            };
            let Some(graph_term) = graph_term else {
                continue;
            };
            for (subject, node) in graph.iter() {
                let Some(node) = node.as_object() else {
                    continue;
                };
                let Some(subject_term) = self.to_named_or_blank(subject) else {
                    continue;
                };
                for (property, values) in node.iter() {
                    let items = match values {
                        JsonValue::Array(items) => items.clone(),
                        JsonValue::Null => Vec::new(),
                        other => vec![other.clone()],
                    };
                    if property == "@type" {
                        for item in items {
                            if let JsonValue::String(type_iri) = item {
                                if let Some(object) = self.to_named_or_blank(&type_iri) {
                                    self.push(
                                        subject_term.clone(),
                                        rdf::TYPE.into_owned(),
                                        object.into(),
                                        &graph_term,
                                    );
                                }
                            }
                        }
                        continue;
                    }
                    if property.starts_with('@') {
                        continue;
                    }
                    let predicate = if let Some(blank) = property.strip_prefix("_:") {
                        if !self.options.produce_generalized_rdf {
                            continue;
                        }
                        NamedOrBlankNode::BlankNode(BlankNode::new_unchecked(blank.to_owned()))
                    } else {
                        match NamedNode::new(property.to_owned()) {
                            Ok(node) => NamedOrBlankNode::NamedNode(node),
                            Err(_) => continue,
                        }
                    };
                    let predicate_node = match predicate {
                        NamedOrBlankNode::NamedNode(n) => n,
                        NamedOrBlankNode::BlankNode(_) => {
                            // Blank predicates are only allowed in generalized RDF, which oxixml's
                            // Quad model cannot represent; skip.
                            continue;
                        }
                    };
                    for item in items {
                        if let Some(object) = self.object_to_rdf(&item, &graph_term) {
                            self.push(
                                subject_term.clone(),
                                predicate_node.clone(),
                                object,
                                &graph_term,
                            );
                        }
                    }
                }
            }
        }
        self.quads
    }

    fn object_to_rdf(&mut self, item: &JsonValue, graph: &GraphName) -> Option<Term> {
        let JsonValue::Object(object) = item else {
            return None;
        };
        // Node object (has @id).
        if let Some(JsonValue::String(id)) = object.get("@id") {
            return self.to_named_or_blank(id).map(Into::into);
        }
        // List object.
        if let Some(list) = object.get("@list") {
            let items = match list {
                JsonValue::Array(items) => items.clone(),
                _ => Vec::new(),
            };
            return Some(self.list_to_rdf(&items, graph));
        }
        // Value object.
        self.value_to_term(object, graph)
    }

    fn value_to_term(&mut self, object: &JsonObject, graph: &GraphName) -> Option<Term> {
        let value = object.get("@value")?;
        let datatype = object.get("@type").and_then(JsonValue::as_str);
        let language = object.get("@language").and_then(JsonValue::as_str);
        let direction = object.get("@direction").and_then(JsonValue::as_str);

        if datatype == Some("@json") {
            let lexical = canonical_json(value);
            return Some(Term::Literal(Literal::new_typed_literal(
                lexical,
                NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#JSON"),
            )));
        }

        let (lexical, resolved_datatype) = match value {
            JsonValue::Bool(b) => (
                if *b {
                    "true".to_owned()
                } else {
                    "false".to_owned()
                },
                datatype
                    .map(str::to_owned)
                    .unwrap_or_else(|| xsd::BOOLEAN.as_str().to_owned()),
            ),
            JsonValue::Number(n) => {
                let (lexical, dt) = number_to_rdf(n, datatype);
                (lexical, dt)
            }
            JsonValue::String(s) => {
                let dt = datatype.map(str::to_owned);
                (s.clone(), dt.unwrap_or_default())
            }
            JsonValue::Null => return None,
            _ => return None,
        };

        // Direction / language handling for strings.
        if matches!(value, JsonValue::String(_)) && datatype.is_none() {
            if let Some(direction) = direction {
                match self.options.rdf_direction {
                    RdfDirection::I18nDatatype => {
                        let lang = language.unwrap_or("").to_ascii_lowercase();
                        let dt = format!("https://www.w3.org/ns/i18n#{lang}_{direction}");
                        if let Ok(dt) = NamedNode::new(dt) {
                            return Some(Term::Literal(Literal::new_typed_literal(lexical, dt)));
                        }
                    }
                    RdfDirection::CompoundLiteral => {
                        return Some(self.compound_literal(&lexical, language, direction, graph));
                    }
                    RdfDirection::None => {}
                }
            }
            if let Some(language) = language {
                // "Only well-formed statements are emitted": an ill-formed language tag drops the
                // whole value rather than silently losing the tag.
                return Literal::new_language_tagged_literal(lexical, language.to_owned())
                    .ok()
                    .map(Term::Literal);
            }
            return Some(Term::Literal(Literal::new_simple_literal(lexical)));
        }

        if resolved_datatype.is_empty() {
            if let Some(language) = language {
                return Literal::new_language_tagged_literal(lexical, language.to_owned())
                    .ok()
                    .map(Term::Literal);
            }
            return Some(Term::Literal(Literal::new_simple_literal(lexical)));
        }
        match NamedNode::new(resolved_datatype) {
            Ok(dt) => Some(Term::Literal(Literal::new_typed_literal(lexical, dt))),
            Err(_) => Some(Term::Literal(Literal::new_simple_literal(lexical))),
        }
    }

    fn compound_literal(
        &mut self,
        lexical: &str,
        language: Option<&str>,
        direction: &str,
        graph: &GraphName,
    ) -> Term {
        let blank = self.fresh_list_node();
        let subject = NamedOrBlankNode::BlankNode(blank.clone());
        self.push(
            subject.clone(),
            rdf::VALUE.into_owned(),
            Term::Literal(Literal::new_simple_literal(lexical.to_owned())),
            graph,
        );
        if let Some(language) = language {
            // The specification normalizes the language tag of a compound literal to lower case.
            self.push(
                subject.clone(),
                NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#language"),
                Term::Literal(Literal::new_simple_literal(language.to_lowercase())),
                graph,
            );
        }
        self.push(
            subject,
            NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#direction"),
            Term::Literal(Literal::new_simple_literal(direction.to_owned())),
            graph,
        );
        Term::from(blank)
    }

    fn list_to_rdf(&mut self, list: &[JsonValue], graph: &GraphName) -> Term {
        if list.is_empty() {
            return rdf::NIL.into();
        }
        let mut bnodes = Vec::with_capacity(list.len());
        for _ in list {
            bnodes.push(self.fresh_list_node());
        }
        for (i, item) in list.iter().enumerate() {
            let Some(subject) = bnodes.get(i).cloned().map(NamedOrBlankNode::BlankNode) else {
                continue;
            };
            // An entry whose object cannot be represented contributes no `rdf:first`, but the
            // list spine (`rdf:rest`) is always emitted.
            if let Some(object) = self.object_to_rdf(item, graph) {
                self.push(subject.clone(), rdf::FIRST.into_owned(), object, graph);
            }
            let rest = match bnodes.get(i + 1) {
                Some(next) => Term::from(next.clone()),
                None => rdf::NIL.into(),
            };
            self.push(subject, rdf::REST.into_owned(), rest, graph);
        }
        bnodes[0].clone().into()
    }

    fn fresh_list_node(&mut self) -> BlankNode {
        let node = BlankNode::new_unchecked(format!("lst{}", self.list_counter));
        self.list_counter += 1;
        node
    }

    fn push(
        &mut self,
        subject: NamedOrBlankNode,
        predicate: NamedNode,
        object: Term,
        graph: &GraphName,
    ) {
        self.quads
            .push(Quad::new(subject, predicate, object, graph.clone()));
    }

    fn to_named_or_blank(&self, value: &str) -> Option<NamedOrBlankNode> {
        if let Some(blank) = value.strip_prefix("_:") {
            Some(NamedOrBlankNode::BlankNode(BlankNode::new_unchecked(
                blank.to_owned(),
            )))
        } else {
            NamedNode::new(value.to_owned())
                .ok()
                .map(NamedOrBlankNode::NamedNode)
        }
    }
}

/// Canonicalizes a JSON number literal into an XSD lexical form and its datatype IRI.
///
/// Implements steps 4 and 5 of [Object to RDF Conversion](https://www.w3.org/TR/json-ld11-api/#object-to-rdf-conversion):
/// a number is written in `xsd:double` form when it has a non-zero fractional part, when its
/// absolute value reaches 10²¹, or when the value object already asks for `xsd:double`; otherwise
/// it is written in `xsd:integer` form. An explicit `@type` is always preserved — only the *lexical
/// form* depends on the number itself.
fn number_to_rdf(literal: &str, datatype: Option<&str>) -> (String, String) {
    let is_plain_integer = {
        let trimmed = literal.strip_prefix('-').unwrap_or(literal);
        !trimmed.is_empty() && trimmed.bytes().all(|b| b.is_ascii_digit())
    };
    let wants_double = datatype == Some(xsd::DOUBLE.as_str());
    if is_plain_integer && !wants_double && literal.len() <= 21 {
        return (
            canonical_integer(literal),
            datatype
                .map(str::to_owned)
                .unwrap_or_else(|| xsd::INTEGER.as_str().to_owned()),
        );
    }
    let Ok(number) = literal.parse::<f64>() else {
        return (
            literal.to_owned(),
            datatype
                .map(str::to_owned)
                .unwrap_or_else(|| xsd::DOUBLE.as_str().to_owned()),
        );
    };
    let as_double =
        wants_double || number.fract() != 0.0 || number.abs() >= 1e21 || !number.is_finite();
    if as_double {
        (
            canonical_double(number),
            datatype
                .map(str::to_owned)
                .unwrap_or_else(|| xsd::DOUBLE.as_str().to_owned()),
        )
    } else {
        (
            format!("{}", number as i128),
            datatype
                .map(str::to_owned)
                .unwrap_or_else(|| xsd::INTEGER.as_str().to_owned()),
        )
    }
}

fn canonical_integer(literal: &str) -> String {
    if let Ok(n) = literal.parse::<i128>() {
        return n.to_string();
    }
    // Big integer: normalize sign and leading zeros manually.
    let (sign, digits) = match literal.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", literal),
    };
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() {
        "0".to_owned()
    } else {
        format!("{sign}{trimmed}")
    }
}

/// Produces the XSD canonical `xsd:double` lexical form of `value`.
#[must_use]
pub fn canonical_double(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-INF" } else { "INF" }.to_owned();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0E0".to_owned()
        } else {
            "0.0E0".to_owned()
        };
    }
    let negative = value < 0.0;
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let mantissa = if mantissa.contains('.') {
        mantissa.to_owned()
    } else {
        format!("{mantissa}.0")
    };
    let exponent: i64 = exponent.parse().unwrap_or(0);
    format!("{}{mantissa}E{exponent}", if negative { "-" } else { "" })
}

/// Produces a canonical JSON serialization (sorted object keys, canonical numbers) for an
/// `@json`-typed literal.
#[must_use]
pub fn canonical_json(value: &JsonValue) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &JsonValue, out: &mut String) {
    match value {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        JsonValue::Number(n) => out.push_str(&canonical_json_number(n)),
        JsonValue::String(s) => write_json_string(s, out),
        JsonValue::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        JsonValue::Object(object) => {
            let mut keys: Vec<&str> = object.keys().collect();
            // RFC 8785 (JCS) sec. 3.2.3: "Property name strings to be sorted are formatted as
            // arrays of UTF-16 code units". That is *not* Rust's `Ord for str` (which compares
            // by code point / UTF-8 byte order): a supplementary-plane key, whose surrogate pair
            // starts at U+D800, must sort before a key in U+E000..=U+FFFF.
            keys.sort_unstable_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_string(key, out);
                out.push(':');
                if let Some(v) = object.get(key) {
                    write_canonical(v, out);
                }
            }
            out.push('}');
        }
    }
}

/// Renders a JSON number the way [JCS](https://www.rfc-editor.org/rfc/rfc8785) requires: the
/// ECMAScript `Number::toString` form of the IEEE-754 double the literal denotes.
fn canonical_json_number(literal: &str) -> String {
    match literal.parse::<f64>() {
        Ok(number) => ecmascript_number(number),
        Err(_) => literal.to_owned(),
    }
}

/// The ECMAScript
/// [`Number::toString`](https://tc39.es/ecma262/#sec-numeric-types-number-tostring) representation
/// of `value`, which is the number format JCS — and therefore `@json` literal canonicalisation —
/// is defined in terms of.
#[must_use]
pub fn ecmascript_number(value: f64) -> String {
    if value == 0.0 {
        return "0".to_owned();
    }
    if !value.is_finite() {
        // Not representable in JSON; render it in a way that at least round-trips as text.
        return format!("{value}");
    }
    let sign = if value < 0.0 { "-" } else { "" };
    // `{:e}` yields the shortest round-tripping decimal in `d.dddde±EE` form.
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .unwrap_or((scientific.as_str(), "0"));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let k = i64::try_from(digits.len()).unwrap_or(i64::MAX);
    let n = exponent.parse::<i64>().unwrap_or(0) + 1;
    let body = if k <= n && n <= 21 {
        let zeros = usize::try_from(n - k).unwrap_or(0);
        format!("{digits}{}", "0".repeat(zeros))
    } else if 0 < n && n <= 21 {
        let split = usize::try_from(n).unwrap_or(0);
        match (digits.get(..split), digits.get(split..)) {
            (Some(head), Some(tail)) => format!("{head}.{tail}"),
            _ => digits.to_owned(),
        }
    } else if -6 < n && n <= 0 {
        let zeros = usize::try_from(-n).unwrap_or(0);
        format!("0.{}{digits}", "0".repeat(zeros))
    } else {
        let exponent = n - 1;
        let exponent_sign = if exponent < 0 { "-" } else { "+" };
        let magnitude = exponent.abs();
        if k == 1 {
            format!("{digits}e{exponent_sign}{magnitude}")
        } else {
            match (digits.get(..1), digits.get(1..)) {
                (Some(head), Some(tail)) => {
                    format!("{head}.{tail}e{exponent_sign}{magnitude}")
                }
                _ => format!("{digits}e{exponent_sign}{magnitude}"),
            }
        }
    };
    format!("{sign}{body}")
}

fn write_json_string(value: &str, out: &mut String) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_double_forms() {
        assert_eq!(canonical_double(1.0), "1.0E0");
        assert_eq!(canonical_double(100.0), "1.0E2");
        assert_eq!(canonical_double(0.001), "1.0E-3");
        assert_eq!(canonical_double(1.1), "1.1E0");
    }

    #[test]
    fn ecmascript_number_forms() {
        assert_eq!(ecmascript_number(0.0), "0");
        assert_eq!(ecmascript_number(42.0), "42");
        assert_eq!(ecmascript_number(100.0), "100");
        assert_eq!(ecmascript_number(4.5), "4.5");
        assert_eq!(ecmascript_number(0.002), "0.002");
        assert_eq!(ecmascript_number(1e30), "1e+30");
        assert_eq!(ecmascript_number(1e-27), "1e-27");
        assert_eq!(ecmascript_number(-2.5), "-2.5");
        assert_eq!(
            ecmascript_number(333_333_333.333_333_3),
            "333333333.3333333"
        );
    }

    /// Regression test for RFC 8785 (JCS) sec. 3.2.3: object members are ordered by UTF-16 code
    /// unit, so a supplementary-plane key sorts *before* one in U+E000..=U+FFFF — the opposite of
    /// Rust's code-point ordering.
    #[test]
    fn canonical_json_sorts_keys_by_utf16_code_unit() -> Result<(), Box<dyn std::error::Error>> {
        let value = JsonValue::from_slice("{\"\u{ffff}\":2,\"\u{10000}\":1}".as_bytes())?;
        assert_eq!(canonical_json(&value), "{\"\u{10000}\":1,\"\u{ffff}\":2}");
        // The RFC's own worked example (sec. 3.2.3).
        let value = JsonValue::from_slice(
            "{\"\u{20ac}\":\"Euro Sign\",\"\\r\":\"Carriage Return\",\"\u{10376}\":\"Ideograph\",\"1\":\"One\"}"
                .as_bytes(),
        )?;
        assert_eq!(
            canonical_json(&value),
            "{\"\\r\":\"Carriage Return\",\"1\":\"One\",\"\u{20ac}\":\"Euro Sign\",\"\u{10376}\":\"Ideograph\"}"
        );
        Ok(())
    }

    #[test]
    fn integer_canonicalization() {
        assert_eq!(number_to_rdf("42", None).0, "42");
        assert_eq!(number_to_rdf("-7", None).0, "-7");
        assert_eq!(number_to_rdf("1.5", None).0, "1.5E0");
    }
}
